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
import secrets
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

ROLES = ["nurse", "provider", "auditor", "admin"]

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
    "GEOGRAPHIC_MISMATCH": "Geographic mismatch (facility outside patient's county)",
    "CLINICAL_INCONSISTENCY": "Clinical inconsistency (condition/gender mismatch)",
}
# One-hot categorical columns (ENCOUNTER_TYPE_wellness, GENDER_F,
# PATIENT_COUNTY_Nairobi, ...) aren't individually listed above - there are
# ~30 of them and they're generated dynamically from the data. Their display
# prefix is handled by pretty_feature_label() below instead.
ONEHOT_PREFIX_LABELS = {
    "ENCOUNTER_TYPE": "Encounter type",
    "GENDER": "Gender",
    "PATIENT_COUNTY": "Patient county",
}


def pretty_feature_label(fname: str) -> str:
    """Human-readable label for a feature column in SHAP output. Checks the
    static FEATURE_LABELS map first (numeric/engineered features), then
    falls back to splitting a one-hot column like PATIENT_COUNTY_Nairobi
    into "Patient county: Nairobi", then falls back to the raw name."""
    if fname in FEATURE_LABELS:
        return FEATURE_LABELS[fname]
    for prefix, label in ONEHOT_PREFIX_LABELS.items():
        if fname.startswith(prefix + "_"):
            value = fname[len(prefix) + 1:]
            return f"{label}: {value}"
    return fname

st.set_page_config(page_title="Kenya Health Claim Fraud Detection", page_icon="\U0001F6E1️", layout="wide")


LIGHT_PALETTE = {
    "bg": "#EEF2F6", "card": "#FBFCFD", "card_border": "#DCE3EA",
    "text": "#1B2A36", "heading": "#14405E", "muted": "#5B7285",
    "sidebar_grad": "linear-gradient(180deg, #123A54 0%, #0B2638 100%)",
    "sidebar_text": "#E8F1F8",
    "input_bg": "#FFFFFF", "input_border": "#C9D3DC",
    "accent": "#14405E",
}
DARK_PALETTE = {
    # Exact spec supplied by the user (their design-system table), not an
    # approximation:
    #   background #0B132B, surface/card #1C2541, primary text #F4F6FA,
    #   secondary/muted text #94A3B8, accent (buttons/links) #00B4D8.
    "bg": "#0B132B", "card": "#1C2541", "card_border": "#2E3B5E",
    "text": "#F4F6FA", "heading": "#F4F6FA", "muted": "#94A3B8",
    "sidebar_grad": "linear-gradient(180deg, #1C2541 0%, #0B132B 100%)",
    "sidebar_text": "#F4F6FA",
    "input_bg": "#1C2541", "input_border": "#2E3B5E",
    "accent": "#00B4D8",
}


def inject_theme(dark: bool = False):
    """Purely cosmetic - a consistent theme (colors, font, card styling) on
    top of Streamlit's defaults, with an optional dark palette. Uses
    data-testid selectors, which are the stable part of Streamlit's DOM
    across recent versions, rather than generated class names, which change
    between releases. Doesn't touch any app logic - safe to skip if a future
    Streamlit version doesn't match a selector, it just falls back to
    default styling for that bit.

    Known limitation: st.dataframe renders through its own internal grid
    component (not plain HTML/CSS), so its cell colors may not fully follow
    this toggle - only the border/corners around it are guaranteed to.
    """
    c = DARK_PALETTE if dark else LIGHT_PALETTE
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        h1, h2, h3 {{ color: {c['heading']}; font-weight: 700; }}
        p, span, label, .stMarkdown, .stCaption {{ color: {c['text']}; }}

        /* Page background - targets stApp/stAppViewContainer AND the newer
           stMain/stMainBlockContainer wrappers Streamlit added in later
           versions, so the canvas paints correctly across versions instead
           of leaving a white/gray panel behind inputs and tables. */
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {{
            background-color: {c['bg']} !important;
        }}

        /* Streamlit's own top toolbar - unstyled by default, so it was
           clashing hard against a dark page (a bright white strip at the
           very top of the app). */
        [data-testid="stHeader"] {{
            background-color: {c['bg']} !important;
        }}
        [data-testid="stHeader"] * {{ color: {c['text']} !important; }}
        [data-testid="stToolbar"] * {{ color: {c['text']} !important; }}

        /* Text/number/date/password inputs - style the actual <input>
           element directly (guaranteed to exist) rather than relying on a
           parent wrapper div, whose exact class/testid varies and wasn't
           matching - that's why these were still showing plain white. */
        input, textarea {{
            background-color: {c['input_bg']} !important;
            color: {c['text']} !important;
            border: 1px solid {c['input_border']} !important;
            border-radius: 8px !important;
        }}
        [data-baseweb="input"], [data-baseweb="base-input"] {{
            background-color: {c['input_bg']} !important;
            border-radius: 8px !important;
        }}
        /* The show/hide-password eye icon button - previously kept its
           default white square background regardless of theme. Scoped to
           the whole stTextInput widget (not just the inner wrapper) since
           the button can sit outside that inner wrapper. */
        [data-testid="stTextInput"] button {{
            background-color: transparent !important;
        }}
        [data-testid="stTextInput"] button svg {{
            fill: {c['muted']} !important;
        }}

        /* Selectboxes */
        [data-baseweb="select"] > div {{
            background-color: {c['input_bg']} !important;
            border-color: {c['input_border']} !important;
        }}
        [data-baseweb="select"] * {{ color: {c['text']} !important; }}
        [data-baseweb="popover"] {{ background-color: {c['input_bg']} !important; }}
        [data-baseweb="popover"] * {{ color: {c['text']} !important; }}
        /* Dropdown option list - rendered in its own portal, not always a
           descendant of the popover div above. */
        [data-baseweb="menu"] {{ background-color: {c['card']} !important; }}
        [data-baseweb="menu"] li, [data-baseweb="menu"] * {{ color: {c['text']} !important; }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
            background: {c['sidebar_grad']};
        }}
        /* Only text-bearing elements go light in the sidebar - NOT buttons,
           which are styled explicitly below. (An earlier version used a
           blanket "* {{ color }}" rule here that also turned button text
           light, making buttons with a light background invisible.) */
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] .stMarkdown {{
            color: {c['sidebar_text']} !important;
        }}
        [data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.15); }}

        /* Sidebar buttons: explicit light-on-dark "ghost button" style so
           they're always visible against the dark sidebar background,
           regardless of light/dark mode. */
        [data-testid="stSidebar"] .stButton button {{
            background-color: rgba(255,255,255,0.12) !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(255,255,255,0.35) !important;
        }}
        [data-testid="stSidebar"] .stButton button:hover {{
            background-color: rgba(255,255,255,0.22) !important;
        }}

        /* Buttons (main content area) */
        .stButton button, [data-testid="stFormSubmitButton"] button {{
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.15s ease;
        }}
        .stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 3px 8px rgba(0,0,0,0.25);
        }}
        /* Primary ("Log in", "Submit claim", etc.) buttons - explicitly tied
           to the accent color so they switch with the toggle. Without this
           they'd fall back to config.toml's static primaryColor, which never
           changes for dark mode. */
        .stButton button[kind="primary"],
        .stButton [data-testid="baseButton-primary"],
        [data-testid="stFormSubmitButton"] button[kind="primary"],
        [data-testid="stFormSubmitButton"] [data-testid="baseButton-primary"] {{
            background-color: {c['accent']} !important;
            border-color: {c['accent']} !important;
            color: {'#0B132B' if dark else '#FFFFFF'} !important;
        }}
        .stButton button[kind="primary"]:hover,
        .stButton [data-testid="baseButton-primary"]:hover,
        [data-testid="stFormSubmitButton"] button[kind="primary"]:hover,
        [data-testid="stFormSubmitButton"] [data-testid="baseButton-primary"]:hover {{
            filter: brightness(1.1);
        }}
        /* Links and other accent-colored text */
        a, a:visited {{ color: {c['accent']} !important; }}

        /* Metric cards */
        [data-testid="stMetric"] {{
            background: {c['card']};
            border: 1px solid {c['card_border']};
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.08);
        }}
        [data-testid="stMetricLabel"] {{ color: {c['muted']}; font-weight: 500; }}
        [data-testid="stMetricValue"] {{ color: {c['heading']}; font-weight: 700; }}

        /* Dataframes / tables - border only; see docstring caveat above */
        [data-testid="stDataFrame"] {{
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid {c['card_border']};
        }}

        /* Form containers (login, claim submission, etc.) get a card look */
        [data-testid="stForm"] {{
            background: {c['card']};
            border: 1px solid {c['card_border']};
            border-radius: 16px;
            padding: 1.6rem 1.8rem;
            box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        }}

        /* Bordered st.container() cards */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: {c['card']};
            border-radius: 16px !important;
            box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        }}
        </style>
    """, unsafe_allow_html=True)


inject_theme(st.session_state.get("dark_mode", False))

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
            salt TEXT,
            role TEXT NOT NULL DEFAULT 'nurse'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            first_name TEXT,
            last_name TEXT,
            gender TEXT,
            dob TEXT,
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
            admission_datetime TEXT,
            discharge_datetime TEXT,
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
    if "salt" not in users_cols:
        # Existing rows get salt = NULL, meaning "still the old unsalted
        # SHA-256 hash" - login() below verifies those the old way once,
        # then transparently re-hashes with a fresh salt on that successful
        # login, so no one's password needs to be reset for this upgrade.
        conn.execute("ALTER TABLE users ADD COLUMN salt TEXT")
        conn.commit()

    # One-time migration: the role previously called "auditor" actually did
    # claim submission (the real mix-up a role review caught - in a real
    # claims system, submission and review have to be different people/
    # accounts). Any account that already has that old role was functionally
    # a submitter, so it becomes "provider" here; "auditor" from this point
    # on means review-only, with no submission access. PRAGMA user_version
    # makes this run exactly once, so an auditor account created normally
    # afterwards (via Manage users) is never touched by it.
    if conn.execute("PRAGMA user_version").fetchone()[0] < 1:
        conn.execute("UPDATE users SET role = 'provider' WHERE role = 'auditor'")
        conn.execute("PRAGMA user_version = 1")
        conn.commit()

    patients_cols = {row[1] for row in conn.execute("PRAGMA table_info(patients)").fetchall()}
    if "dob" not in patients_cols:
        # Existing patients keep their stored (static) age - there's no way
        # to back out a date of birth from a stored age alone. New
        # registrations from here on capture dob and compute age from it
        # automatically instead of asking for age directly.
        conn.execute("ALTER TABLE patients ADD COLUMN dob TEXT")
        conn.commit()

    claims_cols = {row[1] for row in conn.execute("PRAGMA table_info(claims_log)").fetchall()}
    for col, ddl in [
        ("review_status", "ALTER TABLE claims_log ADD COLUMN review_status TEXT DEFAULT 'pending'"),
        ("reviewed_by", "ALTER TABLE claims_log ADD COLUMN reviewed_by TEXT"),
        ("reviewed_at", "ALTER TABLE claims_log ADD COLUMN reviewed_at TEXT"),
        ("patient_id", "ALTER TABLE claims_log ADD COLUMN patient_id TEXT"),
        ("override_reason", "ALTER TABLE claims_log ADD COLUMN override_reason TEXT"),
        # Full admission/discharge timestamps for inpatient claims - NULL
        # for non-inpatient encounters (service_date alone is enough there).
        # length_of_stay_hours is still the feature the model actually uses;
        # these two are the real source it's now computed from for inpatient
        # claims, instead of being typed in as a free-floating number.
        ("admission_datetime", "ALTER TABLE claims_log ADD COLUMN admission_datetime TEXT"),
        ("discharge_datetime", "ALTER TABLE claims_log ADD COLUMN discharge_datetime TEXT"),
    ]:
        if col not in claims_cols:
            conn.execute(ddl)
    conn.commit()

    # Seed demo accounts on first run: admin (full access) and auditor1 (so
    # the auditor role can be demoed under its own account, not just as a
    # subset of admin).
    cur = conn.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        admin_hash, admin_salt = hash_password("admin123")
        conn.execute(
            "INSERT INTO users (username, password_hash, salt, role) VALUES (?, ?, ?, ?)",
            ("admin", admin_hash, admin_salt, "admin"),
        )
        conn.commit()
    return conn


def hash_password(pw: str, salt: str | None = None) -> tuple[str, str]:
    """PBKDF2-HMAC-SHA256 with a random per-user salt (200,000 iterations -
    OWASP's current minimum recommendation for PBKDF2-SHA256). Replaces the
    earlier plain hashlib.sha256(pw) scheme, which had no salt (identical
    passwords produced identical hashes) and was fast enough to be
    brute-forced cheaply if the DB file ever leaked. Returns (hash, salt) -
    pass an existing salt back in to verify a login; omit it to generate a
    fresh one for a new/changed password."""
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200_000).hex()
    return digest, salt


def verify_password(pw: str, stored_hash: str, salt: str | None) -> bool:
    """True if pw matches stored_hash. Handles both the new salted scheme
    (salt is set) and the legacy unsalted SHA-256 scheme (salt is NULL, from
    before this migration) so existing accounts keep working unchanged."""
    if salt:
        check, _ = hash_password(pw, salt)
        return check == stored_hash
    return hashlib.sha256(pw.encode()).hexdigest() == stored_hash


def calculate_age(dob: dt.date, as_of: dt.date | None = None) -> int:
    """Whole years between a date of birth and `as_of` (defaults to today).
    Standard "has this year's birthday happened yet" subtraction, not just
    year difference - avoids being off by one for anyone whose birthday
    hasn't happened yet this year."""
    as_of = as_of or dt.date.today()
    years = as_of.year - dob.year
    if (as_of.month, as_of.day) < (dob.month, dob.day):
        years -= 1
    return max(years, 0)


# ============================================================
# FEATURE ENGINEERING (mirrors export_model_artifacts.py exactly)
# ============================================================
MATERNITY_CONDITIONS = {"Pregnant (finding)", "Pre-eclampsia (disorder)"}


def engineer_row(raw: dict, artifacts: dict, conn) -> dict:
    """raw has the RAW claim fields; returns the full FEATURE_COLUMNS dict."""
    provider_id = str(raw["PROVIDER_ID"])
    condition = raw["PRIMARY_CONDITION"]
    county = raw["PATIENT_COUNTY"]
    claim_amount = raw["CLAIM_AMOUNT_KES"]
    service_date = str(raw["SERVICE_DATE"])
    gender = raw["GENDER"]

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

    # GEOGRAPHIC_MISMATCH: does this provider's facility county differ from
    # the patient's own registered county? (location-anomaly signal)
    provider_county = artifacts.get("provider_county_lookup", {}).get(provider_id)
    geographic_mismatch = int(provider_county is not None and provider_county != county)

    # CLINICAL_INCONSISTENCY: a maternity condition on a male patient
    # (clinical-inconsistency signal)
    clinical_inconsistency = int(str(gender) == "M" and condition in MATERNITY_CONDITIONS)

    feat = {
        "BASE_COST_KES": raw["BASE_COST_KES"],
        "CLAIM_AMOUNT_KES": claim_amount,
        "PAYER_COVERAGE_KES": raw["PAYER_COVERAGE_KES"],
        "CLAIM_TO_BASE_RATIO": raw["CLAIM_TO_BASE_RATIO"],
        "COST_RATIO": cost_ratio,
        "PROVIDER_Z_SCORE": z_score,
        "PROVIDER_DAILY_VOLUME": daily_volume,
        "AGE": raw["AGE"],
        "LENGTH_OF_STAY_HOURS": raw["LENGTH_OF_STAY_HOURS"],
        "GEOGRAPHIC_MISMATCH": geographic_mismatch,
        "CLINICAL_INCONSISTENCY": clinical_inconsistency,
    }

    # One-hot encode the three categorical fields exactly the way
    # export_model_artifacts.py built them (pd.get_dummies, prefix=col) -
    # every known category gets its own 0/1 column, matching the model's
    # trained feature set. An unseen value (shouldn't happen - these come
    # from dropdowns populated from artifacts["onehot_categories"]) just
    # leaves every dummy column for that field at 0, which XGBoost treats
    # the same way it treats an all-missing one-hot group.
    raw_categorical_values = {
        "ENCOUNTER_TYPE": raw["ENCOUNTER_TYPE"],
        "GENDER": raw["GENDER"],
        "PATIENT_COUNTY": county,
    }
    for col, categories in artifacts["onehot_categories"].items():
        value = str(raw_categorical_values[col])
        for cat in categories:
            feat[f"{col}_{cat}"] = 1 if value == cat else 0

    return feat


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
            admission_datetime, discharge_datetime,
            age, length_of_stay_hours, base_cost_kes, claim_amount_kes,
            payer_coverage_kes, claim_to_base_ratio, fraud_score, risk_label,
            review_status
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        dt.datetime.now().isoformat(timespec="seconds"), username, source,
        raw.get("PATIENT_ID"), str(raw["PROVIDER_ID"]), raw["PRIMARY_CONDITION"],
        raw["ENCOUNTER_TYPE"], raw["GENDER"], raw["PATIENT_COUNTY"], str(raw["SERVICE_DATE"]),
        raw.get("ADMISSION_DATETIME"), raw.get("DISCHARGE_DATETIME"),
        raw["AGE"], raw["LENGTH_OF_STAY_HOURS"],
        raw["BASE_COST_KES"], raw["CLAIM_AMOUNT_KES"], raw["PAYER_COVERAGE_KES"],
        raw["CLAIM_TO_BASE_RATIO"], score, label, "pending",
    ))
    conn.commit()


# ============================================================
# LOGIN
# ============================================================
def login_screen(conn):
    top_l, top_r = st.columns([5, 1])
    with top_r:
        st.toggle("\U0001F319 Dark mode", key="dark_mode")

    col_l, col_mid, col_r = st.columns([1, 1.3, 1])
    with col_mid:
        st.markdown(
            "<div style='text-align:center; font-size:2.6rem; margin-bottom:0.2rem;'>"
            "\U0001F6E1️</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<h1 style='text-align:center; font-size:1.6rem; margin-bottom:0.2rem;'>"
            "Kenya Health Claim Fraud Detection</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:center; color:#5B7285; margin-bottom:1.6rem;'>"
            "An Explainable Healthcare Claim Fraud Detection System for "
            "Kenyan Insurance Companies</p>",
            unsafe_allow_html=True,
        )
        with st.form("login_form"):
            st.markdown("**Sign in**")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
        if submitted:
            cur = conn.execute("SELECT password_hash, salt, role FROM users WHERE username = ?", (username,))
            row = cur.fetchone()
            if row and verify_password(password, row[0], row[1]):
                if row[1] is None:
                    # Legacy unsalted account that just verified correctly -
                    # transparently upgrade it to a salted hash now that we
                    # have the plaintext password in hand.
                    new_hash, new_salt = hash_password(password)
                    conn.execute(
                        "UPDATE users SET password_hash = ?, salt = ? WHERE username = ?",
                        (new_hash, new_salt, username),
                    )
                    conn.commit()
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.role = row[2]
                st.rerun()
            else:
                st.error("Invalid username or password.")
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
            dob = st.date_input(
                "Date of birth", value=dt.date(1996, 1, 1),
                min_value=dt.date(1900, 1, 1), max_value=dt.date.today(),
            )
        county = st.selectbox("County", artifacts["counties"])
        submitted = st.form_submit_button("Register patient", type="primary")

    if submitted:
        if not first_name.strip() or not last_name.strip():
            st.error("First and last name are both required.")
        elif dob > dt.date.today():
            st.error("Date of birth can't be in the future.")
        else:
            age = calculate_age(dob)
            count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
            patient_id = f"P-{88200 + count + 1}"
            conn.execute(
                "INSERT INTO patients (patient_id, first_name, last_name, gender, dob, age, county, "
                "registered_by, registered_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (patient_id, first_name.strip(), last_name.strip(), gender, dob.isoformat(), age, county,
                 st.session_state.username, dt.datetime.now().isoformat(timespec="seconds")),
            )
            conn.commit()
            st.success("✓ Patient registered successfully")
            st.info(f"Patient ID: **{patient_id}**  ·  age **{age}** (calculated from date of birth)  "
                    "—  hand this to the patient / use it when submitting their claims")

    st.divider()
    st.subheader("Recently registered")
    df = pd.read_sql_query(
        "SELECT patient_id AS \"Patient ID\", first_name || ' ' || last_name AS Name, "
        "gender AS Gender, dob AS dob_raw, age AS age_fallback, "
        "county AS County FROM patients ORDER BY registered_at DESC LIMIT 20",
        conn,
    )
    if df.empty:
        st.caption("No patients registered yet.")
    else:
        # Live age, recomputed from date of birth on every page load (not
        # the static value stored at registration time) - patients registered
        # before this column existed have no dob on file, so they fall back
        # to their stored (static) age instead.
        def _live_age(row):
            if row["dob_raw"]:
                return calculate_age(dt.date.fromisoformat(row["dob_raw"]))
            return row["age_fallback"]

        df["Age"] = df.apply(_live_age, axis=1)
        df["Date of birth"] = df["dob_raw"].fillna("— not on file —")
        display_cols = ["Patient ID", "Name", "Gender", "Date of birth", "Age", "County"]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)


# ============================================================
# CLAIM SUBMISSION (MANUAL FORM)
# ============================================================
def submit_claim_page(conn, model, artifacts):
    st.header("Submit a claim for scoring")
    fd = artifacts["form_defaults"]

    patients_df = pd.read_sql_query(
        "SELECT patient_id, first_name, last_name, dob, age FROM patients "
        "ORDER BY registered_at DESC LIMIT 300",
        conn,
    )

    # Patient selection lives OUTSIDE the form below, not by accident -
    # widgets inside st.form() only report their value on submit, so an
    # auto-filled age tied to patient selection has to react immediately,
    # before the form is submitted.
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

    # Auto-calculated from the linked patient's date of birth, not typed in -
    # falls back to a manual field only for a walk-in with no patient record,
    # or a legacy patient registered before dob was captured.
    linked_age = None
    if patient_id is not None:
        p = patients_df[patients_df["patient_id"] == patient_id].iloc[0]
        if p["dob"]:
            linked_age = calculate_age(dt.date.fromisoformat(p["dob"]))
        else:
            linked_age = int(p["age"]) if p["age"] is not None else None

    # Encounter type also lives outside the form - whether this is an
    # inpatient claim decides which fields render below (admission/discharge
    # date+time vs. a single length-of-stay number), and that choice has to
    # take effect immediately, same reason patient selection is out here.
    encounter_type = st.selectbox("Encounter type", artifacts["encounter_types"])
    is_inpatient = encounter_type == "inpatient"

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
            gender = st.selectbox("Gender", artifacts["genders"])
        with col2:
            admission_dt = discharge_dt = None
            if is_inpatient:
                st.caption("Inpatient - admission and discharge date/time (length of stay "
                           "is calculated from these, not entered directly).")
                ac1, ac2 = st.columns(2)
                admission_date = ac1.date_input("Admission date", value=dt.date.today())
                admission_time = ac2.time_input("Admission time", value=dt.time(9, 0))
                dc1, dc2 = st.columns(2)
                discharge_date = dc1.date_input("Discharge date", value=dt.date.today())
                discharge_time = dc2.time_input("Discharge time", value=dt.time(9, 0))
                admission_dt = dt.datetime.combine(admission_date, admission_time)
                discharge_dt = dt.datetime.combine(discharge_date, discharge_time)
                service_date = admission_date
                length_of_stay = max((discharge_dt - admission_dt).total_seconds() / 3600, 0.0)
                st.metric("Length of stay (hours)", round(length_of_stay, 1))
            else:
                service_date = st.date_input("Service date", value=dt.date.today())
                length_of_stay = st.number_input(
                    "Length of stay (hours)", min_value=0.0,
                    value=round(fd["LENGTH_OF_STAY_HOURS"]["median"], 2),
                )
            if linked_age is not None:
                st.metric("Patient age", linked_age)
                st.caption("Calculated from the linked patient's date of birth.")
                age = linked_age
            else:
                age = st.number_input(
                    "Patient age", min_value=0, max_value=120, value=int(fd["AGE"]["median"]),
                    help="No date of birth on file for this patient (walk-in, or registered "
                         "before date-of-birth capture) - enter age manually.",
                )
        with col3:
            base_cost = st.number_input("Base cost (KES)", min_value=0.0, value=round(fd["BASE_COST_KES"]["median"], 2))
            claim_amount = st.number_input("Claim amount (KES)", min_value=0.0, value=round(fd["CLAIM_AMOUNT_KES"]["median"], 2))
            payer_coverage = st.number_input("Payer coverage (KES)", min_value=0.0, value=round(fd["PAYER_COVERAGE_KES"]["median"], 2))
            claim_to_base_ratio = round(claim_amount / base_cost, 3) if base_cost else 1.0
            st.metric("Claim-to-base ratio", claim_to_base_ratio)

        submitted = st.form_submit_button("Score this claim", type="primary")

    if submitted and is_inpatient and discharge_dt < admission_dt:
        st.error("Discharge date/time can't be before admission date/time.")
    elif submitted:
        raw = {
            "PATIENT_ID": patient_id, "PROVIDER_ID": provider_id, "PRIMARY_CONDITION": condition,
            "PATIENT_COUNTY": county, "ENCOUNTER_TYPE": encounter_type, "GENDER": gender,
            "SERVICE_DATE": service_date, "AGE": age,
            "LENGTH_OF_STAY_HOURS": length_of_stay, "BASE_COST_KES": base_cost,
            "CLAIM_AMOUNT_KES": claim_amount, "PAYER_COVERAGE_KES": payer_coverage,
            "CLAIM_TO_BASE_RATIO": claim_to_base_ratio,
            "ADMISSION_DATETIME": admission_dt.isoformat() if admission_dt else None,
            "DISCHARGE_DATETIME": discharge_dt.isoformat() if discharge_dt else None,
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
# CLAIM DETAIL - SCORING & DECISION
# Opened by clicking a row in the Claims dashboard. Matches wireframe 4.
# The score/SHAP explanation are visible to both provider and auditor (a
# provider can see why their own submission was flagged); the actual
# approve/reject/escalate decision is auditor-only, enforced below by
# `role`, not just by hiding the nav entry.
# ============================================================
def claim_detail_page(conn, model, artifacts, claim_id, role):
    if st.button("← Back to Claims dashboard"):
        st.session_state.selected_claim_id = None
        st.rerun()

    row = conn.execute(
        "SELECT id, submitted_at, submitted_by, source, patient_id, provider_id, primary_condition, "
        "encounter_type, gender, county, service_date, admission_datetime, discharge_datetime, "
        "age, length_of_stay_hours, base_cost_kes, "
        "claim_amount_kes, payer_coverage_kes, claim_to_base_ratio, fraud_score, risk_label, "
        "review_status, reviewed_by, reviewed_at, override_reason FROM claims_log WHERE id = ?",
        (claim_id,),
    ).fetchone()
    if row is None:
        st.error("Claim not found.")
        return

    (cid, submitted_at, submitted_by, source, patient_id, provider_id, condition, encounter_type,
     gender, county, service_date, admission_datetime, discharge_datetime, age, los, base_cost,
     claim_amount, payer_coverage,
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
        if admission_datetime and discharge_datetime:
            st.write(f"**Admission:** {admission_datetime}")
            st.write(f"**Discharge:** {discharge_datetime}")
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
                    fname_label = pretty_feature_label(fname)
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

    if role != "auditor":
        # Providers can see the score, explanation, and the review status
        # above, but cannot act on it - only an independent auditor account
        # can approve/reject/escalate a claim. This is enforced here, not
        # just by hiding "Submit claim"/"Audit history" from the provider
        # nav, so a provider can't reach this control by any other path.
        st.caption(
            "Only an auditor account can approve, reject, or escalate a claim - "
            "providers can view the outcome here once an auditor has reviewed it."
        )
        return

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
# FRAUD ANALYTICS OVERVIEW (auditor)
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
               "nurse -> register patients only; provider -> submit/upload claims, "
               "claims dashboard, provider profile (this is the hospital/facility "
               "staff who generate a claim); auditor -> review claims (approve/"
               "reject/escalate), audit history, analytics, provider profiles - "
               "review-only, no submission access, so the same account can't both "
               "submit and approve a claim. Admin is scoped strictly to provisioning "
               "accounts (this page only) - it is not a superset with access to "
               "claims or clinical data.")

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
                new_hash, new_salt = hash_password(new_password)
                conn.execute(
                    "INSERT INTO users (username, password_hash, salt, role) VALUES (?, ?, ?, ?)",
                    (new_username, new_hash, new_salt, new_role),
                )
                conn.commit()
                st.success(f"User '{new_username}' created with role '{new_role}'. They can log in now.")

    st.divider()
    st.subheader("Existing users")
    users_df = pd.read_sql_query("SELECT username, role FROM users ORDER BY username", conn)
    st.dataframe(users_df, use_container_width=True)

    admin_count = int((users_df["role"] == "admin").sum())

    st.divider()
    st.subheader("Edit a user's role")
    if users_df.empty:
        st.caption("No users yet.")
    else:
        edit_username = st.selectbox("Select a user", users_df["username"].tolist(), key="edit_role_user")
        current_role = users_df.loc[users_df["username"] == edit_username, "role"].iloc[0]
        st.caption(f"Current role: **{current_role}**")
        new_role_for_edit = st.selectbox(
            "New role", ROLES,
            index=ROLES.index(current_role) if current_role in ROLES else 0,
            key="edit_role_value",
        )
        if st.button("Update role", type="primary"):
            if current_role == "admin" and new_role_for_edit != "admin" and admin_count <= 1:
                st.error("Can't change this account's role - it's the only remaining admin. "
                         "Create another admin account first if you want to demote this one.")
            elif new_role_for_edit == current_role:
                st.info("No change - that's already this account's role.")
            else:
                conn.execute("UPDATE users SET role = ? WHERE username = ?", (new_role_for_edit, edit_username))
                conn.commit()
                st.success(f"'{edit_username}' role changed from '{current_role}' to '{new_role_for_edit}'.")
                st.rerun()

    st.divider()
    st.subheader("Remove a user")
    admin_usernames = set(users_df.loc[users_df["role"] == "admin", "username"])
    if admin_count <= 1:
        removable = [u for u in users_df["username"] if u not in admin_usernames]
    else:
        removable = list(users_df["username"])
    if removable:
        to_remove = st.selectbox("Select a user to remove", removable, key="remove_user_select")
        if st.button("Remove user", type="secondary"):
            conn.execute("DELETE FROM users WHERE username = ?", (to_remove,))
            conn.commit()
            st.success(f"User '{to_remove}' removed.")
            st.rerun()
    else:
        st.caption("No removable users (can't remove the only remaining admin account).")


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
    st.sidebar.toggle("\U0001F319 Dark mode", key="dark_mode")
    st.sidebar.divider()

    nav_options = []
    if role == "nurse":
        nav_options += ["Register patient"]
    if role == "provider":
        # Provider = hospital/facility staff who generate and submit claims.
        # They can see their own submissions' status in Claims dashboard, but
        # claim_detail_page below hides the approve/reject/escalate controls
        # unless the viewer is an auditor - providers can't review their own
        # submissions, only auditors can.
        nav_options += ["Submit claim", "Upload CSV", "Claims dashboard", "Provider profile"]
    if role == "auditor":
        # Auditor = independent reviewer. No submission access - review-only,
        # by design, so the same account can't both submit and approve a claim.
        nav_options += ["Claims dashboard", "Audit history", "Analytics", "Provider profile"]
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
            claim_detail_page(conn, model, artifacts, st.session_state.selected_claim_id, role)
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
