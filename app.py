"""
App task: Prototype 1 - Basic web interface with XGBoost fraud scoring.
Login -> role-based navigation (nurse: register patients only, the
front-line clinical actor; auditor: submit/upload/review claims, dashboard,
analytics - the actor who actually handles claims per the use case; admin:
account provisioning ONLY - not a superset, no access to claims or clinical
data) -> colour-coded score. Pages and navigation follow the
supervisor-approved wireframes (Fraud_Detection_System_Wireframes_1.drawio).

Serves predictions from the v2-robust XGBoost model (missingness-trained,
regularized, adaptive dual-threshold triage), using lookups exported by
export_model_artifacts.py so engineered features match training exactly.
Local SQLite (claims_dashboard_v2.db) stores users, registered patients, and
a log of every claim scored through the app.

Claim Detail shows a real per-claim SHAP explanation (shap.TreeExplainer on
the live XGBoost model) rather than a canned one. The "Hybrid Score" row in
the wireframe (XGBoost + GraphSAGE) is left as an explicit "not yet
integrated" note until GraphSAGE (Prototype 2) exists - showing a fabricated
number there would be worse than saying it plainly isn't ready yet.
"""

import os
import json
import sqlite3
import hashlib
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

BASE_DIR = os.path.expanduser("~/synthea_kenya")
MODEL_PATH = f"{BASE_DIR}/output/model/xgboost_fraud_model_v2_robust.json"
ARTIFACTS_PATH = f"{BASE_DIR}/output/model/dashboard_artifacts.json"
DB_PATH = f"{BASE_DIR}/output/model/claims_dashboard_v2.db"

ROLES = ["nurse", "auditor", "admin"]

FEATURE_LABELS = {
    "BASE_COST_KES": "Base cost (KES)",
    "CLAIM_AMOUNT_KES": "Claim amount (KES)",
    "PAYER_COVERAGE_KES": "Payer coverage (KES)",
    "CLAIM_TO_BASE_RATIO": "Claim-to-base ratio",
    "COST_RATIO": "Cost ratio vs. similar claims",
    "PROVIDER_Z_SCORE": "Provider cost z-score",
    "PROVIDER_DAILY_VOLUME": "Provider daily claim volume",
    "AGE": "Patient age",
    "LENGTH_OF_STAY_HOURS": "Length of stay (hours)",
    "ENCOUNTER_TYPE_ENCODED": "Encounter type",
    "GENDER_ENCODED": "Gender",
    "PATIENT_COUNTY_ENCODED": "Patient county",
}

st.set_page_config(page_title="Kenya Health Claim Fraud Detection", page_icon="\U0001F6E1️", layout="wide")

# ============================================================
# CACHED LOADERS
# ============================================================
@st.cache_resource
def load_model():
    m = xgb.XGBClassifier()
    m.load_model(MODEL_PATH)
    return m


@st.cache_resource
def load_artifacts():
    with open(ARTIFACTS_PATH) as f:
        return json.load(f)


@st.cache_resource
def load_shap_explainer(_model):
    # Leading underscore tells st.cache_resource not to hash this arg (the
    # model object) - it's the same cached singleton from load_model() every
    # call anyway. TreeExplainer needs no background dataset for XGBoost.
    if not SHAP_AVAILABLE:
        return None
    return shap.TreeExplainer(_model)


def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    # The connected folder is a FUSE-style mount that doesn't support SQLite's
    # normal POSIX file locking (default journal mode fails with "disk I/O
    # error" there). This app is single-process/single-user locally, so an
    # in-memory rollback journal with synchronous writes off is safe and
    # avoids needing filesystem locks at all.
    conn.execute("PRAGMA journal_mode=MEMORY")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'nurse'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            first_name TEXT,
            last_name TEXT,
            gender TEXT,
            age INTEGER,
            county TEXT,
            registered_by TEXT,
            registered_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS claims_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            submitted_at TEXT,
            submitted_by TEXT,
            source TEXT,
            patient_id TEXT,
            provider_id TEXT,
            primary_condition TEXT,
            encounter_type TEXT,
            gender TEXT,
            county TEXT,
            service_date TEXT,
            age REAL,
            length_of_stay_hours REAL,
            base_cost_kes REAL,
            claim_amount_kes REAL,
            payer_coverage_kes REAL,
            claim_to_base_ratio REAL,
            fraud_score REAL,
            risk_label TEXT,
            review_status TEXT DEFAULT 'pending',
            reviewed_by TEXT,
            reviewed_at TEXT,
            override_reason TEXT
        )
    """)
    conn.commit()

    # Migration: CREATE TABLE IF NOT EXISTS doesn't add columns to a table
    # that already exists on disk from before these columns were added, so
    # an already-running dashboard's DB file needs this on next launch.
    users_cols = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "role" not in users_cols:
        conn.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'nurse'")
        conn.execute("UPDATE users SET role = 'admin' WHERE username = 'admin'")
        conn.commit()

    claims_cols = {row[1] for row in conn.execute("PRAGMA table_info(claims_log)").fetchall()}
    for col, ddl in [
        ("review_status", "ALTER TABLE claims_log ADD COLUMN review_status TEXT DEFAULT 'pending'"),
        ("reviewed_by", "ALTER TABLE claims_log ADD COLUMN reviewed_by TEXT"),
        ("reviewed_at", "ALTER TABLE claims_log ADD COLUMN reviewed_at TEXT"),
        ("patient_id", "ALTER TABLE claims_log ADD COLUMN patient_id TEXT"),
        ("override_reason", "ALTER TABLE claims_log ADD COLUMN override_reason TEXT"),
    ]:
        if col not in claims_cols:
            conn.execute(ddl)
    conn.commit()

    # Seed demo accounts on first run: admin (full access) and auditor1 (so
    # the auditor role can be demoed under its own account, not just as a
    # subset of admin).
    cur = conn.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            ("admin", hash_password("admin123"), "admin"),
        )
        conn.commit()
    return conn


def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


# ============================================================
# FEATURE ENGINEERING (mirrors export_model_artifacts.py exactly)
# ============================================================
def engineer_row(raw: dict, artifacts: dict, conn) -> dict:
    """raw has the RAW claim fields; returns the full FEATURE_COLUMNS dict."""
    provider_id = str(raw["PROVIDER_ID"])
    condition = raw["PRIMARY_CONDITION"]
    county = raw["PATIENT_COUNTY"]
    claim_amount = raw["CLAIM_AMOUNT_KES"]
    service_date = str(raw["SERVICE_DATE"])

    # COST_RATIO
    key = f"{condition}||{county}"
    median = artifacts["group_median_lookup"].get(key, artifacts["global_median"])
    cost_ratio = (claim_amount / median) if median else 1.0

    # PROVIDER_Z_SCORE
    p_mean = artifacts["provider_mean_lookup"].get(provider_id, artifacts["global_mean"])
    p_std = artifacts["provider_std_lookup"].get(provider_id, artifacts["global_std"]) or artifacts["global_std"]
    z_score = (claim_amount - p_mean) / p_std if p_std else 0.0

    # PROVIDER_DAILY_VOLUME: historical baseline + claims already logged
    # through this app for the same provider/date (including this one)
    baseline = artifacts["provider_avg_daily_volume"].get(provider_id, artifacts["global_avg_daily_volume"])
    cur = conn.execute(
        "SELECT COUNT(*) FROM claims_log WHERE provider_id = ? AND service_date = ?",
        (provider_id, service_date),
    )
    today_count = cur.fetchone()[0]
    daily_volume = baseline + today_count + 1

    def encode(col, value):
        classes = artifacts["label_encoders"][col]
        return classes.index(value) if value in classes else 0

    return {
        "BASE_COST_KES": raw["BASE_COST_KES"],
        "CLAIM_AMOUNT_KES": claim_amount,
        "PAYER_COVERAGE_KES": raw["PAYER_COVERAGE_KES"],
        "CLAIM_TO_BASE_RATIO": raw["CLAIM_TO_BASE_RATIO"],
        "COST_RATIO": cost_ratio,
        "PROVIDER_Z_SCORE": z_score,
        "PROVIDER_DAILY_VOLUME": daily_volume,
        "AGE": raw["AGE"],
        "LENGTH_OF_STAY_HOURS": raw["LENGTH_OF_STAY_HOURS"],
        "ENCOUNTER_TYPE_ENCODED": encode("ENCOUNTER_TYPE", raw["ENCOUNTER_TYPE"]),
        "GENDER_ENCODED": encode("GENDER", raw["GENDER"]),
        "PATIENT_COUNTY_ENCODED": encode("PATIENT_COUNTY", county),
    }


def score_row(feat: dict, model, artifacts) -> tuple:
    X = pd.DataFrame([feat])[artifacts["feature_columns"]]
    proba = float(model.predict_proba(X)[:, 1][0])
    return proba, risk_label(proba, artifacts)


def risk_label(proba: float, artifacts: dict) -> str:
    if proba >= artifacts["t_high"]:
        return "HIGH"
    elif proba >= artifacts["t_low"]:
        return "MEDIUM"
    return "LOW"


def label_badge(label: str) -> str:
    return {"HIGH": "\U0001F534 HIGH RISK – auto-flag for fraud review",
            "MEDIUM": "\U0001F7E1 MEDIUM RISK – route to manual audit",
            "LOW": "\U0001F7E2 LOW RISK – clear"}[label]


def log_claim(conn, raw, source, username, score, label):
    conn.execute("""
        INSERT INTO claims_log (
            submitted_at, submitted_by, source, patient_id, provider_id,
            primary_condition, encounter_type, gender, county, service_date,
            age, length_of_stay_hours, base_cost_kes, claim_amount_kes,
            payer_coverage_kes, claim_to_base_ratio, fraud_score, risk_label,
            review_status
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        dt.datetime.now().isoformat(timespec="seconds"), username, source,
        raw.get("PATIENT_ID"), str(raw["PROVIDER_ID"]), raw["PRIMARY_CONDITION"],
        raw["ENCOUNTER_TYPE"], raw["GENDER"], raw["PATIENT_COUNTY"], str(raw["SERVICE_DATE"]),
        raw["AGE"], raw["LENGTH_OF_STAY_HOURS"],
        raw["BASE_COST_KES"], raw["CLAIM_AMOUNT_KES"], raw["PAYER_COVERAGE_KES"],
        raw["CLAIM_TO_BASE_RATIO"], score, label, "pending",
    ))
    conn.commit()


# ============================================================
# LOGIN
# ============================================================
def login_screen(conn):
    st.title("\U0001F6E1️ Kenya Health Claim Fraud Detection")
    st.caption("An Explainable Healthcare Claim Fraud Detection System for Kenyan Insurance Companies")
    st.subheader("Sign in")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
    if submitted:
        cur = conn.execute("SELECT password_hash, role FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if row and row[0] == hash_password(password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = row[1]
            st.rerun()
        else:
            st.error("Invalid username or password.")
    st.info(
        "Demo accounts: **admin** / **admin123** (role: admin), "
        "**auditor1** / **auditor123** (role: auditor). "
        "Role is set per account (see Manage users), not chosen at login - "
        "that's deliberate, letting someone pick their own access level at "
        "sign-in would defeat the point of having roles on a fraud system."
    )
    st.caption("Forgot password? Contact your System Admin.")


# ============================================================
# REGISTER PATIENT (nurse / admin)
# ============================================================
def register_patient_page(conn, artifacts):
    st.header("Register patient")
    with st.form("register_patient_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            first_name = st.text_input("First name")
            gender = st.selectbox("Gender", artifacts.get("genders", ["Female", "Male"]))
        with c2:
            last_name = st.text_input("Last name")
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
        county = st.selectbox("County", artifacts["counties"])
        submitted = st.form_submit_button("Register patient", type="primary")

    if submitted:
        if not first_name.strip() or not last_name.strip():
            st.error("First and last name are both required.")
        else:
            count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
            patient_id = f"P-{88200 + count + 1}"
            conn.execute(
                "INSERT INTO patients (patient_id, first_name, last_name, gender, age, county, "
                "registered_by, registered_at) VALUES (?,?,?,?,?,?,?,?)",
                (patient_id, first_name.strip(), last_name.strip(), gender, age, county,
                 st.session_state.username, dt.datetime.now().isoformat(timespec="seconds")),
            )
            conn.commit()
            st.success("✓ Patient registered successfully")
            st.info(f"Patient ID: **{patient_id}**  —  hand this to the patient / use it when submitting their claims")

    st.divider()
    st.subheader("Recently registered")
    df = pd.read_sql_query(
        "SELECT patient_id AS \"Patient ID\", first_name || ' ' || last_name AS Name, "
        "gender AS Gender, age AS Age, county AS County FROM patients "
        "ORDER BY registered_at DESC LIMIT 20",
        conn,
    )
    if df.empty:
        st.caption("No patients registered yet.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


# ============================================================
# CLAIM SUBMISSION (MANUAL FORM)
# ============================================================
def submit_claim_page(conn, model, artifacts):
    st.header("Submit a claim for scoring")
    fd = artifacts["form_defaults"]

    patients_df = pd.read_sql_query(
        "SELECT patient_id, first_name, last_name FROM patients ORDER BY registered_at DESC LIMIT 300",
        conn,
    )

    with st.form("claim_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            if patients_df.empty:
                st.caption("No patients registered yet - register one on the Register patient "
                           "page first if you want to link this claim to a patient (optional).")
                patient_id = None
            else:
                patient_options = [None] + patients_df["patient_id"].tolist()

                def _patient_label(pid):
                    if pid is None:
                        return "— no patient linked (walk-in / not yet registered) —"
                    r = patients_df[patients_df["patient_id"] == pid].iloc[0]
                    return f"{pid} — {r['first_name']} {r['last_name']}"

                patient_id = st.selectbox("Patient", patient_options, format_func=_patient_label)

            org_names = artifacts.get("organization_names", {})
            org_lookup = artifacts["provider_org_lookup"]

            def _provider_label(pid: str) -> str:
                org_id = org_lookup.get(pid, "unknown")
                name = org_names.get(org_id, f"Facility {org_id[:8]}")
                return f"{name}  ·  Provider {pid[:8]}"

            provider_id = st.selectbox(
                "Provider", artifacts["top_providers"], format_func=_provider_label
            )
            org_id = org_lookup.get(provider_id, "unknown")
            org_name = org_names.get(org_id, "unknown facility")
            st.caption(f"Facility: {org_name}")
            condition = st.selectbox("Primary condition", artifacts["conditions"])
            county = st.selectbox("Patient county", artifacts["counties"])
            encounter_type = st.selectbox("Encounter type", artifacts["encounter_types"])
            gender = st.selectbox("Gender", artifacts["genders"])
        with col2:
            service_date = st.date_input("Service date", value=dt.date.today())
            age = st.number_input("Patient age", min_value=0, max_value=120, value=int(fd["AGE"]["median"]))
            length_of_stay = st.number_input("Length of stay (hours)", min_value=0.0, value=round(fd["LENGTH_OF_STAY_HOURS"]["median"], 2))
        with col3:
            base_cost = st.number_input("Base cost (KES)", min_value=0.0, value=round(fd["BASE_COST_KES"]["median"], 2))
            claim_amount = st.number_input("Claim amount (KES)", min_value=0.0, value=round(fd["CLAIM_AMOUNT_KES"]["median"], 2))
            payer_coverage = st.number_input("Payer coverage (KES)", min_value=0.0, value=round(fd["PAYER_COVERAGE_KES"]["median"], 2))
            claim_to_base_ratio = round(claim_amount / base_cost, 3) if base_cost else 1.0
            st.metric("Claim-to-base ratio", claim_to_base_ratio)

        submitted = st.form_submit_button("Score this claim", type="primary")

    if submitted:
        raw = {
            "PATIENT_ID": patient_id, "PROVIDER_ID": provider_id, "PRIMARY_CONDITION": condition,
            "PATIENT_COUNTY": county, "ENCOUNTER_TYPE": encounter_type, "GENDER": gender,
            "SERVICE_DATE": service_date, "AGE": age,
            "LENGTH_OF_STAY_HOURS": length_of_stay, "BASE_COST_KES": base_cost,
            "CLAIM_AMOUNT_KES": claim_amount, "PAYER_COVERAGE_KES": payer_coverage,
            "CLAIM_TO_BASE_RATIO": claim_to_base_ratio,
        }
        feat = engineer_row(raw, artifacts, conn)
        score, label = score_row(feat, model, artifacts)
        log_claim(conn, raw, "manual", st.session_state.username, score, label)

        st.divider()
        st.subheader("Result")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Fraud score", f"{score:.3f}")
        with c2:
            if label == "HIGH":
                st.error(label_badge(label))
            elif label == "MEDIUM":
                st.warning(label_badge(label))
            else:
                st.success(label_badge(label))
        st.caption(
            f"Thresholds: score < {artifacts['t_low']:.3f} = low risk, "
            f"{artifacts['t_low']:.3f}–{artifacts['t_high']:.3f} = medium (manual review), "
            f">= {artifacts['t_high']:.3f} = high (auto-flag)."
        )


# ============================================================
# CSV UPLOAD (BATCH SCORING)
# ============================================================
REQUIRED_CSV_COLUMNS = [
    "PROVIDER_ID", "PRIMARY_CONDITION", "PATIENT_COUNTY", "ENCOUNTER_TYPE",
    "GENDER", "SERVICE_DATE", "AGE", "LENGTH_OF_STAY_HOURS",
    "BASE_COST_KES", "CLAIM_AMOUNT_KES", "PAYER_COVERAGE_KES", "CLAIM_TO_BASE_RATIO",
]


def upload_csv_page(conn, model, artifacts):
    st.header("Batch-score claims from a CSV")
    st.caption("Required columns: " + ", ".join(REQUIRED_CSV_COLUMNS) + ". Optional: PATIENT_ID.")

    template = pd.DataFrame([{
        "PATIENT_ID": "",
        "PROVIDER_ID": artifacts["top_providers"][0],
        "PRIMARY_CONDITION": artifacts["conditions"][0],
        "PATIENT_COUNTY": artifacts["counties"][0],
        "ENCOUNTER_TYPE": artifacts["encounter_types"][0],
        "GENDER": artifacts["genders"][0],
        "SERVICE_DATE": dt.date.today().isoformat(),
        "AGE": 40, "LENGTH_OF_STAY_HOURS": 2,
        "BASE_COST_KES": 2000, "CLAIM_AMOUNT_KES": 3500,
        "PAYER_COVERAGE_KES": 2500, "CLAIM_TO_BASE_RATIO": 1.75,
    }])
    st.download_button("Download CSV template", template.to_csv(index=False), "claim_template.csv")

    uploaded = st.file_uploader("Upload claims CSV", type="csv")
    if uploaded is None:
        return

    df = pd.read_csv(uploaded)
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    st.write(f"{len(df)} claims loaded. Scoring...")
    scores, labels = [], []
    for _, row in df.iterrows():
        raw = row[REQUIRED_CSV_COLUMNS].to_dict()
        raw["PATIENT_ID"] = row.get("PATIENT_ID") if "PATIENT_ID" in df.columns and pd.notna(row.get("PATIENT_ID")) else None
        feat = engineer_row(raw, artifacts, conn)
        score, label = score_row(feat, model, artifacts)
        log_claim(conn, raw, "csv", st.session_state.username, score, label)
        scores.append(score)
        labels.append(label)

    df["FRAUD_SCORE"] = scores
    df["RISK_LABEL"] = labels

    def highlight(row):
        color = {"HIGH": "#ffcccc", "MEDIUM": "#fff3cd", "LOW": "#d4edda"}[row["RISK_LABEL"]]
        return [f"background-color: {color}"] * len(row)

    st.dataframe(df.style.apply(highlight, axis=1), use_container_width=True)
    st.download_button("Download scored results", df.to_csv(index=False), "scored_claims.csv")

    counts = df["RISK_LABEL"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("High risk", int(counts.get("HIGH", 0)))
    c2.metric("Medium risk", int(counts.get("MEDIUM", 0)))
    c3.metric("Low risk", int(counts.get("LOW", 0)))


# ============================================================
# AUDIT HISTORY
# ============================================================
def history_page(conn):
    st.header("Audit history")
    df = pd.read_sql_query(
        "SELECT submitted_at, submitted_by, source, patient_id, provider_id, encounter_type, "
        "claim_amount_kes, fraud_score, risk_label, review_status FROM claims_log "
        "ORDER BY id DESC LIMIT 500",
        conn,
    )
    if df.empty:
        st.info("No claims scored yet.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Total scored", len(df))
    c2.metric("High risk", int((df["risk_label"] == "HIGH").sum()))
    c3.metric("Medium risk", int((df["risk_label"] == "MEDIUM").sum()))
    st.dataframe(df, use_container_width=True)


# ============================================================
# CLAIMS DASHBOARD (auditor / admin) - search, filter, click a row
# to open Claim Detail. Matches wireframe 2 ("Claims Dashboard").
# ============================================================
def claims_dashboard_page(conn, artifacts):
    st.header("Claims dashboard")

    df = pd.read_sql_query(
        "SELECT id, submitted_at, patient_id, provider_id, primary_condition, "
        "claim_amount_kes, fraud_score, risk_label, review_status FROM claims_log "
        "ORDER BY id DESC LIMIT 1000",
        conn,
    )
    if df.empty:
        st.info("No claims scored yet.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Total claims", len(df))
    c2.metric("High risk", int((df["risk_label"] == "HIGH").sum()))
    c3.metric("Pending review", int((df["review_status"] == "pending").sum()))

    search_col, risk_col, status_col = st.columns([2, 1, 1])
    search = search_col.text_input("Search claims (claim ID, patient ID, provider ID)")
    risk_filter = risk_col.selectbox("Risk level", ["All", "HIGH", "MEDIUM", "LOW"])
    status_filter = status_col.selectbox("Status", ["All", "pending", "approved", "rejected", "escalated"])

    org_names = artifacts.get("organization_names", {})
    org_lookup = artifacts["provider_org_lookup"]
    df["facility"] = df["provider_id"].map(lambda pid: org_names.get(org_lookup.get(pid, ""), "unknown facility"))

    filtered = df
    if search.strip():
        s = search.strip().lower()
        mask = (
            filtered["id"].astype(str).str.contains(s)
            | filtered["patient_id"].fillna("").astype(str).str.lower().str.contains(s)
            | filtered["provider_id"].astype(str).str.lower().str.contains(s)
        )
        filtered = filtered[mask]
    if risk_filter != "All":
        filtered = filtered[filtered["risk_label"] == risk_filter]
    if status_filter != "All":
        filtered = filtered[filtered["review_status"] == status_filter]

    st.caption(f"{len(filtered)} claim(s). Click a row to open its Claim Detail screen.")

    display_df = filtered[[
        "id", "patient_id", "facility", "primary_condition", "claim_amount_kes",
        "fraud_score", "risk_label", "review_status",
    ]].rename(columns={
        "id": "Claim ID", "patient_id": "Patient", "facility": "Provider",
        "primary_condition": "Diagnosis", "claim_amount_kes": "Amount (KES)",
        "fraud_score": "Risk score", "risk_label": "Risk label", "review_status": "Status",
    })

    event = st.dataframe(
        display_df, use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row",
    )
    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        st.session_state.selected_claim_id = int(filtered.iloc[selected_rows[0]]["id"])
        st.rerun()


# ============================================================
# CLAIM DETAIL - SCORING & DECISION (auditor / admin)
# Opened by clicking a row in the Claims dashboard. Matches wireframe 4.
# ============================================================
def claim_detail_page(conn, model, artifacts, claim_id):
    if st.button("← Back to Claims dashboard"):
        st.session_state.selected_claim_id = None
        st.rerun()

    row = conn.execute(
        "SELECT id, submitted_at, submitted_by, source, patient_id, provider_id, primary_condition, "
        "encounter_type, gender, county, service_date, age, length_of_stay_hours, base_cost_kes, "
        "claim_amount_kes, payer_coverage_kes, claim_to_base_ratio, fraud_score, risk_label, "
        "review_status, reviewed_by, reviewed_at, override_reason FROM claims_log WHERE id = ?",
        (claim_id,),
    ).fetchone()
    if row is None:
        st.error("Claim not found.")
        return

    (cid, submitted_at, submitted_by, source, patient_id, provider_id, condition, encounter_type,
     gender, county, service_date, age, los, base_cost, claim_amount, payer_coverage,
     ratio, score, label, review_status, reviewed_by, reviewed_at, override_reason) = row

    st.header(f"Claim #{cid}")

    patient_label = "— none linked —"
    if patient_id:
        p = conn.execute(
            "SELECT first_name, last_name, gender, age, county FROM patients WHERE patient_id = ?",
            (patient_id,),
        ).fetchone()
        patient_label = f"{patient_id}  ({p[0]} {p[1]}, {p[2]}, {p[3]}, {p[4]})" if p else patient_id

    org_names = artifacts.get("organization_names", {})
    org_lookup = artifacts["provider_org_lookup"]
    org_id = org_lookup.get(provider_id, "unknown")
    facility = org_names.get(org_id, f"Facility {org_id[:8]}")

    col_left, col_right = st.columns([1, 1.3])
    with col_left:
        st.subheader("Claim details")
        st.write(f"**Claim ID:** {cid}")
        st.write(f"**Patient:** {patient_label}")
        st.write(f"**Provider:** {facility}  ·  Provider {provider_id[:8]}")
        st.write(f"**Encounter date:** {service_date}")
        st.write(f"**Diagnosis:** {condition}")
        st.write(f"**Encounter type:** {encounter_type}")
        st.write(f"**Gender:** {gender}")
        st.write(f"**County:** {county}")
        st.write(f"**Patient age (at claim):** {age:.0f}")
        st.write(f"**Length of stay:** {los:.1f} hours")
        st.write(f"**Base cost (SHA tariff):** KES {base_cost:,.0f}")
        st.write(f"**Claim amount:** KES {claim_amount:,.0f}")
        st.write(f"**Payer coverage:** KES {payer_coverage:,.0f}")
        st.write(f"**Claim-to-base ratio:** {ratio:.2f}")
        st.caption(f"Submitted by {submitted_by} ({source}) on {submitted_at}")

    with col_right:
        st.subheader("Fraud risk score")
        sc1, sc2 = st.columns([1, 2])
        with sc1:
            st.metric("Score", f"{score:.3f}")
        with sc2:
            if label == "HIGH":
                st.error(label_badge(label))
            elif label == "MEDIUM":
                st.warning(label_badge(label))
            else:
                st.success(label_badge(label))
        st.caption(
            f"XGBoost: {score:.3f}  ·  GraphSAGE: not yet integrated (Prototype 2, in progress)  ·  "
            "Hybrid score: not available until GraphSAGE is trained - using XGBoost only for now."
        )

        st.subheader("Explanation (SHAP)")
        if not SHAP_AVAILABLE:
            st.caption(
                "The `shap` package isn't installed in this environment, so the "
                "per-claim explanation can't be computed right now. Run "
                "`pip install shap` (in the same terminal you launch this app "
                "from) and restart the app to enable it - this doesn't affect "
                "scoring, only this explanation panel."
            )
        else:
            raw = {
                "PROVIDER_ID": provider_id, "PRIMARY_CONDITION": condition, "PATIENT_COUNTY": county,
                "ENCOUNTER_TYPE": encounter_type, "GENDER": gender, "SERVICE_DATE": service_date,
                "AGE": age, "LENGTH_OF_STAY_HOURS": los, "BASE_COST_KES": base_cost,
                "CLAIM_AMOUNT_KES": claim_amount, "PAYER_COVERAGE_KES": payer_coverage,
                "CLAIM_TO_BASE_RATIO": ratio,
            }
            try:
                feat = engineer_row(raw, artifacts, conn)
                X_row = pd.DataFrame([feat])[artifacts["feature_columns"]]
                explainer = load_shap_explainer(model)
                sv = np.array(explainer.shap_values(X_row)).reshape(-1)
                contrib = sorted(
                    zip(artifacts["feature_columns"], sv), key=lambda t: abs(t[1]), reverse=True
                )[:3]
                for fname, val in contrib:
                    fname_label = FEATURE_LABELS.get(fname, fname)
                    direction = "increases" if val > 0 else "decreases"
                    st.write(f"- **{fname_label}** ({feat[fname]:.2f}) {direction} the fraud score ({val:+.3f})")
            except Exception as e:
                st.caption(f"SHAP explanation unavailable: {e}")

    st.divider()
    st.subheader("Auditor decision")
    if review_status != "pending":
        note = f"Already reviewed: **{review_status}** by {reviewed_by} on {reviewed_at}"
        if override_reason:
            note += f"  ·  override reason: {override_reason}"
        st.info(note)

    override = st.checkbox("Override model verdict")
    reason = st.text_input("Override reason (required if overriding)") if override else ""

    b1, b2, b3 = st.columns(3)
    decision = None
    if b1.button("Approve", type="primary", use_container_width=True):
        decision = "approved"
    if b2.button("Reject", use_container_width=True):
        decision = "rejected"
    if b3.button("Escalate", use_container_width=True):
        decision = "escalated"

    if decision:
        if override and not reason.strip():
            st.error("An override reason is required when overriding the model verdict.")
        else:
            conn.execute(
                "UPDATE claims_log SET review_status=?, reviewed_by=?, reviewed_at=?, override_reason=? "
                "WHERE id=?",
                (
                    decision, st.session_state.username,
                    dt.datetime.now().isoformat(timespec="seconds"),
                    reason.strip() if override else None, cid,
                ),
            )
            conn.commit()
            st.success(f"Claim #{cid} marked '{decision}'.")
            st.rerun()


# ============================================================
# FRAUD ANALYTICS OVERVIEW (auditor / admin)
# ============================================================
def analytics_page(conn):
    st.header("Fraud analytics overview")
    df = pd.read_sql_query(
        "SELECT submitted_at, provider_id, fraud_score, risk_label FROM claims_log",
        conn,
    )
    if df.empty:
        st.info("No claims scored yet - analytics will populate as claims come in "
                "through Submit claim / Upload CSV.")
        return

    total = len(df)
    flagged = int(df["risk_label"].isin(["HIGH", "MEDIUM"]).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total claims scored", total)
    c2.metric("Flagged (High+Medium)", flagged)
    c3.metric("Flag rate", f"{flagged / total * 100:.1f}%")
    c4.metric("Avg fraud score", f"{df['fraud_score'].mean():.3f}")

    st.subheader("Risk label breakdown")
    counts = df["risk_label"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"]).fillna(0)
    st.bar_chart(counts)

    st.subheader("Claims scored per day")
    df["date"] = pd.to_datetime(df["submitted_at"]).dt.date
    daily = df.groupby("date").size()
    st.line_chart(daily)

    st.subheader("Top 10 flagged providers (by flagged claim count)")
    flagged_df = df[df["risk_label"].isin(["HIGH", "MEDIUM"])]
    if flagged_df.empty:
        st.caption("No flagged claims yet.")
    else:
        top = flagged_df["provider_id"].value_counts().head(10)
        st.bar_chart(top)


# ============================================================
# PROVIDER RISK PROFILE (auditor / admin)
# ============================================================
def provider_profile_page(conn, artifacts):
    st.header("Provider risk profile")

    org_names = artifacts.get("organization_names", {})
    org_lookup = artifacts["provider_org_lookup"]

    def _provider_label(pid: str) -> str:
        org_id = org_lookup.get(pid, "unknown")
        name = org_names.get(org_id, f"Facility {org_id[:8]}")
        return f"{name}  ·  Provider {pid[:8]}"

    provider_id = st.selectbox(
        "Provider", artifacts["top_providers"], format_func=_provider_label
    )
    org_id = org_lookup.get(provider_id, "unknown")
    org_name = org_names.get(org_id, "unknown facility")
    st.caption(f"Facility: {org_name}")

    st.subheader("Historical baseline (from training data)")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Avg claim amount (KES)",
        f"{artifacts['provider_mean_lookup'].get(provider_id, artifacts['global_mean']):,.0f}",
    )
    c2.metric(
        "Claim amount std dev",
        f"{artifacts['provider_std_lookup'].get(provider_id, artifacts['global_std']):,.0f}",
    )
    c3.metric(
        "Avg daily claim volume",
        f"{artifacts['provider_avg_daily_volume'].get(provider_id, artifacts['global_avg_daily_volume']):.1f}",
    )

    st.divider()
    st.subheader("Claims scored through this app")
    df = pd.read_sql_query(
        "SELECT submitted_at, primary_condition, claim_amount_kes, fraud_score, "
        "risk_label, review_status FROM claims_log WHERE provider_id = ? ORDER BY id DESC",
        conn, params=(provider_id,),
    )
    if df.empty:
        st.info("No claims logged for this provider yet through the dashboard.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Claims logged", len(df))
    c2.metric("Avg fraud score", f"{df['fraud_score'].mean():.3f}")
    c3.metric("Flagged", int(df["risk_label"].isin(["HIGH", "MEDIUM"]).sum()))
    st.dataframe(df, use_container_width=True)


# ============================================================
# MANAGE USERS (admin only)
# ============================================================
def manage_users_page(conn):
    st.header("Manage users")
    st.caption("New accounts are saved to the users table in the SQLite database and "
               "can log in immediately. Role controls which pages an account can see: "
               "nurse -> register patients only; auditor -> submit/upload claims, "
               "claims dashboard, audit history, analytics, provider profiles. "
               "Admin is scoped strictly to provisioning accounts (this page only) - "
               "it is not a superset with access to claims or clinical data.")

    with st.form("add_user_form", clear_on_submit=True):
        new_username = st.text_input("New username")
        new_role = st.selectbox("Role", ROLES, index=0)
        new_password = st.text_input("New password", type="password")
        confirm_password = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Add user", type="primary")

    if submitted:
        if not new_username or not new_password:
            st.error("Username and password are both required.")
        elif new_password != confirm_password:
            st.error("Passwords don't match.")
        else:
            existing = conn.execute(
                "SELECT 1 FROM users WHERE username = ?", (new_username,)
            ).fetchone()
            if existing:
                st.error(f"Username '{new_username}' already exists.")
            else:
                conn.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                    (new_username, hash_password(new_password), new_role),
                )
                conn.commit()
                st.success(f"User '{new_username}' created with role '{new_role}'. They can log in now.")

    st.divider()
    st.subheader("Existing users")
    users_df = pd.read_sql_query("SELECT username, role FROM users ORDER BY username", conn)
    st.dataframe(users_df, use_container_width=True)

    st.divider()
    st.subheader("Remove a user")
    removable = [u for u in users_df["username"] if u != "admin"]
    if removable:
        to_remove = st.selectbox("Select a user to remove", removable)
        if st.button("Remove user", type="secondary"):
            conn.execute("DELETE FROM users WHERE username = ?", (to_remove,))
            conn.commit()
            st.success(f"User '{to_remove}' removed.")
            st.rerun()
    else:
        st.caption("No removable users (the admin account can't be removed here).")


# ============================================================
# MAIN
# ============================================================
def main():
    conn = get_db()
    if not st.session_state.get("logged_in"):
        login_screen(conn)
        return

    model = load_model()
    artifacts = load_artifacts()
    role = st.session_state.get("role", "nurse")

    st.sidebar.title("\U0001F6E1️ Fraud Detection")
    st.sidebar.caption(f"Signed in as **{st.session_state.username}**  ·  role: **{role}**")

    nav_options = []
    if role == "nurse":
        nav_options += ["Register patient"]
    if role == "auditor":
        nav_options += [
            "Submit claim", "Upload CSV", "Claims dashboard", "Audit history",
            "Analytics", "Provider profile",
        ]
    if role == "admin":
        # Admin is scoped strictly to account provisioning - no claims access,
        # no clinical/audit pages. It is not a superset of nurse/auditor.
        nav_options += ["Manage users"]

    page = st.sidebar.radio("Navigate", nav_options)
    if page != "Claims dashboard":
        st.session_state.selected_claim_id = None

    if st.sidebar.button("Log out"):
        st.session_state.logged_in = False
        st.rerun()

    if page == "Register patient":
        register_patient_page(conn, artifacts)
    elif page == "Submit claim":
        submit_claim_page(conn, model, artifacts)
    elif page == "Upload CSV":
        upload_csv_page(conn, model, artifacts)
    elif page == "Claims dashboard":
        if st.session_state.get("selected_claim_id"):
            claim_detail_page(conn, model, artifacts, st.session_state.selected_claim_id)
        else:
            claims_dashboard_page(conn, artifacts)
    elif page == "Audit history":
        history_page(conn)
    elif page == "Analytics":
        analytics_page(conn)
    elif page == "Provider profile":
        provider_profile_page(conn, artifacts)
    else:
        manage_users_page(conn)


if __name__ == "__main__":
    main()
