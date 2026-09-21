"""
App task: Prototype 1 - Basic web interface with XGBoost fraud scoring.
Login -> claim submission (manual form + CSV upload) -> colour-coded score.

Serves predictions from the v2-robust XGBoost model (missingness-trained,
regularized, adaptive dual-threshold triage), using lookups exported by
export_model_artifacts.py so engineered features match training exactly.
Local SQLite (claims_dashboard_v2.db) stores users and a log of every claim
scored through the app.
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

BASE_DIR = os.path.expanduser("~/synthea_kenya")
MODEL_PATH = f"{BASE_DIR}/output/model/xgboost_fraud_model_v2_robust.json"
ARTIFACTS_PATH = f"{BASE_DIR}/output/model/dashboard_artifacts.json"
DB_PATH = f"{BASE_DIR}/output/model/claims_dashboard_v2.db"

st.set_page_config(page_title="Kenya Health Claim Fraud Detection", page_icon="\U0001F6E1️", layout="wide")

# ============================================================
# CACHED LOADERS
# ===========================================================
@st.cache_resource
def load_model():
    m = xgb.XGBClassifier()
    m.load_model(MODEL_PATH)
    return m


@st.cache_resource
def load_artifacts():
    with open(ARTIFACTS_PATH) as f:
        return json.load(f)


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
            password_hash TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS claims_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            submitted_at TEXT,
            submitted_by TEXT,
            source TEXT,
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
            risk_label TEXT
        )
    """)
    conn.commit()
    # Seed one demo account on first run
    cur = conn.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            ("admin", hash_password("admin123")),
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
            submitted_at, submitted_by, source, provider_id, primary_condition,
            encounter_type, gender, county, service_date, age,
            length_of_stay_hours, base_cost_kes, claim_amount_kes,
            payer_coverage_kes, claim_to_base_ratio, fraud_score, risk_label
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        dt.datetime.now().isoformat(timespec="seconds"), username, source,
        str(raw["PROVIDER_ID"]), raw["PRIMARY_CONDITION"], raw["ENCOUNTER_TYPE"],
        raw["GENDER"], raw["PATIENT_COUNTY"], str(raw["SERVICE_DATE"]),
        raw["AGE"], raw["LENGTH_OF_STAY_HOURS"],
        raw["BASE_COST_KES"], raw["CLAIM_AMOUNT_KES"], raw["PAYER_COVERAGE_KES"],
        raw["CLAIM_TO_BASE_RATIO"], score, label,
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
        cur = conn.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if row and row[0] == hash_password(password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password.")
    st.info("Demo account: **admin** / **admin123**")


# ============================================================
# CLAIM SUBMISSION (MANUAL FORM)
# ============================================================
def submit_claim_page(conn, model, artifacts):
    st.header("Submit a claim for scoring")
    fd = artifacts["form_defaults"]

    with st.form("claim_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
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
            "PROVIDER_ID": provider_id, "PRIMARY_CONDITION": condition,
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
    st.caption("Required columns: " + ", ".join(REQUIRED_CSV_COLUMNS))

    template = pd.DataFrame([{
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
# HISTORY
# ============================================================
def history_page(conn):
    st.header("Claim scoring history")
    df = pd.read_sql_query(
        "SELECT submitted_at, submitted_by, source, provider_id, encounter_type, "
        "claim_amount_kes, fraud_score, risk_label FROM claims_log ORDER BY id DESC LIMIT 500",
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
# MANAGE USERS (admin only)
# ============================================================
def manage_users_page(conn):
    st.header("Manage users")
    st.caption("New accounts are saved to the users table in the SQLite database "
               "and can log in immediately.")

    with st.form("add_user_form", clear_on_submit=True):
        new_username = st.text_input("New username")
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
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (new_username, hash_password(new_password)),
                )
                conn.commit()
                st.success(f"User '{new_username}' created. They can log in now.")

    st.divider()
    st.subheader("Existing users")
    users_df = pd.read_sql_query("SELECT username FROM users ORDER BY username", conn)
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

    st.sidebar.title("\U0001F6E1️ Fraud Detection")
    st.sidebar.caption(f"Signed in as **{st.session_state.username}**")
    nav_options = ["Submit claim", "Upload CSV", "History"]
    if st.session_state.username == "admin":
        nav_options.append("Manage users")
    page = st.sidebar.radio("Navigate", nav_options)
    if st.sidebar.button("Log out"):
        st.session_state.logged_in = False
        st.rerun()

    if page == "Submit claim":
        submit_claim_page(conn, model, artifacts)
    elif page == "Upload CSV":
        upload_csv_page(conn, model, artifacts)
    elif page == "History":
        history_page(conn)
    else:
        manage_users_page(conn)


if __name__ == "__main__":
    main()
