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
the live XGBoost model) rather than a canned one.

HYBRID SCORING (XGBoost + GraphSAGE, Prototype 2 now wired in): every scored
claim also gets a GraphSAGE "relational fraud" score from
graphsage_artifacts.json (precomputed per-node embeddings + the tiny edge-MLP
head's weights, applied here by hand with numpy - no torch/torch_geometric
dependency in the live app). GraphSAGE specialises in collusion_ring-style
fraud (a dense, repeating patient-provider clique) that looks financially
unremarkable row-by-row, which is exactly the fraud type XGBoost alone
struggles with (see tune_hybrid_threshold.py). The final label escalates to
HIGH whenever EITHER model's own tuned threshold fires, so neither model's
blind spot becomes the system's blind spot - this mirrors the OR-rule
validated in hybrid_ensemble_eval.py / tune_hybrid_threshold.py. A brand new
patient or provider with no claims in the training graph has no GraphSAGE
embedding yet (there's no relational history to measure), so the app falls
back to XGBoost alone for them and says so plainly, rather than fabricating
a relational score with nothing behind it.
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

import claim_reference as cref

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

BASE_DIR = os.path.expanduser("~/synthea_kenya")
MODEL_PATH = f"{BASE_DIR}/output/model/xgboost_fraud_model_v2_robust.json"
ARTIFACTS_PATH = f"{BASE_DIR}/output/model/dashboard_artifacts.json"
GRAPHSAGE_ARTIFACTS_PATH = f"{BASE_DIR}/output/model/graphsage_artifacts.json"
GRAPH_NEIGHBORHOOD_PATH = f"{BASE_DIR}/output/model/graph_neighborhood.json"
DB_PATH = f"{BASE_DIR}/output/model/claims_dashboard_v2.db"
PER_FRAUDTYPE_RECALL_PATH = f"{BASE_DIR}/output/model/per_fraudtype_recall_v2data.csv"
HYBRID_ROBUSTNESS_PATH = f"{BASE_DIR}/output/model/hybrid_robustness_results.csv"
HYBRID_THRESHOLDS_PATH = f"{BASE_DIR}/output/model/hybrid_tuned_thresholds.txt"

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
def load_graphsage_artifacts():
    """Returns None (not an error) if the file isn't there yet - lets the
    app run in XGBoost-only mode on an older model export rather than
    crashing, same spirit as the SHAP_AVAILABLE fallback above."""
    if not os.path.exists(GRAPHSAGE_ARTIFACTS_PATH):
        return None
    with open(GRAPHSAGE_ARTIFACTS_PATH) as f:
        return json.load(f)


@st.cache_resource
def load_graph_neighborhood():
    """Compact patient/provider adjacency (export_graph_neighborhood.py) for
    the Claim Detail relational view. None if not exported yet - the page
    just skips the graph rather than failing."""
    if not os.path.exists(GRAPH_NEIGHBORHOOD_PATH):
        return None
    with open(GRAPH_NEIGHBORHOOD_PATH) as f:
        return json.load(f)


def build_neighborhood_dot(patient_id, provider_id, nb, gs_artifacts, org_names, org_lookup):
    """Graphviz DOT for this claim's local patient<->provider neighbourhood.
    Nodes: the claim's patient (blue) and provider (dark), the patient's other
    providers, and the provider's other frequent patients. Edge width = claim
    count; an edge is red when GraphSAGE's own score for that pair is at/above
    its auto-flag threshold, so the picture shows what the model reacts to,
    and a ring shows up as a dense red mesh. Shown edges are limited to the
    top neighbours exported, so counts are 'top partners', not exhaustive."""
    pat_top = nb["patient_top_providers"]
    prov_top = nb["provider_top_patients"]
    thr = gs_artifacts["t_graphsage"] if gs_artifacts else None
    patient_id, provider_id = str(patient_id), str(provider_id)

    def prov_label(pid):
        return org_names.get(org_lookup.get(pid, ""), f"Provider {pid[:6]}")

    shown_patients = {patient_id}
    shown_providers = {provider_id}
    for pat, _ in prov_top.get(provider_id, [])[:12]:
        shown_patients.add(pat)
    for prov, _ in pat_top.get(patient_id, [])[:6]:
        shown_providers.add(prov)

    edges = {}
    for pat in shown_patients:
        for prov, n in pat_top.get(pat, []):
            if prov in shown_providers:
                edges[(pat, prov)] = n
    for prov in shown_providers:
        for pat, n in prov_top.get(prov, []):
            if pat in shown_patients:
                edges.setdefault((pat, prov), n)

    lines = ["graph G {", "rankdir=LR; overlap=false; splines=true;",
             'node [fontname="Helvetica", fontsize=10, style=filled, penwidth=0];']
    for pat in shown_patients:
        is_c = pat == patient_id
        lines.append(
            f'"P:{pat}" [label="{"THIS PATIENT" if is_c else "Patient " + pat[:6]}", shape=ellipse, '
            f'fillcolor="{"#1f77b4" if is_c else "#aec7e8"}", fontcolor="{"white" if is_c else "black"}"];'
        )
    for prov in shown_providers:
        is_c = prov == provider_id
        label = prov_label(prov).replace('"', "'")
        lines.append(
            f'"R:{prov}" [label="{"THIS PROVIDER: " if is_c else ""}{label}", shape=box, '
            f'fillcolor="{"#333333" if is_c else "#d9d9d9"}", fontcolor="{"white" if is_c else "black"}"];'
        )
    for (pat, prov), n in edges.items():
        gs = graphsage_ring_score(pat, prov, gs_artifacts)
        hot = gs is not None and thr is not None and gs >= thr
        width = min(1 + n ** 0.5 / 2, 6)
        color = "#d62728" if hot else "#999999"
        lines.append(f'"P:{pat}" -- "R:{prov}" [penwidth={width:.1f}, color="{color}", label="{n}", fontsize=8];')
    lines.append("}")
    return "\n".join(lines), len(shown_patients), len(shown_providers), len(edges)


@st.cache_resource
def load_shap_explainer(_model):
    # Leading underscore tells st.cache_resource not to hash this arg (the
    # model object) - it's the same cached singleton from load_model() every
    # call anyway. TreeExplainer needs no background dataset for XGBoost.
    if not SHAP_AVAILABLE:
        return None
    return shap.TreeExplainer(_model)


@st.cache_data
def load_model_health_artifacts():
    """Read-only reports from the training/evaluation scripts (per-fraud-type
    recall, the hybrid robustness stress test, tuned thresholds) - not
    recomputed here, just surfaced for the Model Health page. Returns a dict
    with None for anything not found on disk, rather than erroring, since
    this page should still partially render for an older model export that's
    missing one of the newer report files (e.g. hybrid_robustness_results.csv
    didn't exist until this session's stress-test run)."""
    out = {"per_fraudtype_recall": None, "hybrid_robustness": None, "t_xgb": None, "t_graphsage": None}
    if os.path.exists(PER_FRAUDTYPE_RECALL_PATH):
        out["per_fraudtype_recall"] = pd.read_csv(PER_FRAUDTYPE_RECALL_PATH)
    if os.path.exists(HYBRID_ROBUSTNESS_PATH):
        out["hybrid_robustness"] = pd.read_csv(HYBRID_ROBUSTNESS_PATH)
    if os.path.exists(HYBRID_THRESHOLDS_PATH):
        with open(HYBRID_THRESHOLDS_PATH) as f:
            for line in f:
                if line.startswith("t_xgb="):
                    out["t_xgb"] = float(line.strip().split("=")[1])
                elif line.startswith("t_graphsage="):
                    out["t_graphsage"] = float(line.strip().split("=")[1])
    return out


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
            registered_at TEXT,
            scheme TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS encounters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            provider_id TEXT,
            recorded_by TEXT,
            recorded_at TEXT,
            encounter_type TEXT,
            primary_condition TEXT,
            start_datetime TEXT,
            end_datetime TEXT,
            ward TEXT,
            notes TEXT,
            status TEXT DEFAULT 'open',
            claim_id INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS encounter_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            encounter_id INTEGER NOT NULL,
            item TEXT,
            qty REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS claim_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            claim_id INTEGER NOT NULL,
            category TEXT,
            item TEXT,
            qty REAL,
            tariff_price REAL,
            billed_price REAL
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
    # Append-only audit trail per claim: every auditor decision and every
    # investigator note, with actor + timestamp. claims_log only keeps the
    # LAST review state, so without this a claim that is escalated then
    # approved loses its history.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS claim_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            claim_id INTEGER NOT NULL,
            created_at TEXT,
            actor TEXT,
            event_type TEXT,
            detail TEXT
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

    if "scheme" not in patients_cols:
        conn.execute("ALTER TABLE patients ADD COLUMN scheme TEXT")
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
        # GraphSAGE's relational-fraud score (NULL for claims scored before
        # Prototype 2 was wired in, or for a patient/provider with no graph
        # history yet) and which model actually triggered the final label.
        ("graphsage_score", "ALTER TABLE claims_log ADD COLUMN graphsage_score REAL"),
        ("flagged_by", "ALTER TABLE claims_log ADD COLUMN flagged_by TEXT"),
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


def graphsage_ring_score(patient_id, provider_id, gs_artifacts: dict | None):
    """The GraphSAGE encoder + edge-MLP head, applied by hand with numpy
    from precomputed per-node embeddings (see export_graphsage_artifacts.py)
    - no torch/torch_geometric dependency in the live app. Returns None
    (not 0.0 - a real "no opinion", not a vote of confidence) when either
    node has no embedding, i.e. a brand new patient or provider with no
    claims yet in the training graph: there's genuinely no relational
    history to measure yet, which is a realistic limitation of a graph
    model, not a bug to hide."""
    if gs_artifacts is None or patient_id is None or provider_id is None:
        return None
    pe = gs_artifacts["patient_embeddings"].get(str(patient_id))
    pre = gs_artifacts["provider_embeddings"].get(str(provider_id))
    if pe is None or pre is None:
        return None
    w = gs_artifacts["edge_mlp_weights"]
    e = np.concatenate([np.array(pe), np.array(pre)])
    h1 = np.maximum(0.0, np.array(w["W1"]) @ e + np.array(w["b1"]))  # ReLU (dropout is a no-op at eval)
    logit = float((np.array(w["W2"]) @ h1 + np.array(w["b2"])).reshape(-1)[0])
    return 1.0 / (1.0 + np.exp(-logit))


def hybrid_label(xgb_proba: float, gs_score, artifacts: dict, gs_artifacts: dict | None) -> tuple[str, str]:
    """Combines the two models with an OR rule: escalate to HIGH whenever
    EITHER model's own tuned threshold fires (validated in
    hybrid_ensemble_eval.py / tune_hybrid_threshold.py - the two models'
    false positives barely overlap, so ORing them costs almost nothing in
    precision while closing XGBoost's collusion_ring blind spot). Returns
    (label, flagged_by) - flagged_by is shown to the auditor so they know
    WHICH signal fired, not just that something did."""
    xgb_lab = risk_label(xgb_proba, artifacts)
    gs_threshold = gs_artifacts["t_graphsage"] if gs_artifacts else None
    gs_flag = gs_score is not None and gs_threshold is not None and gs_score >= gs_threshold
    if gs_flag and xgb_lab == "HIGH":
        return "HIGH", "both models (cost anomaly + relational pattern)"
    if gs_flag:
        return "HIGH", "relational pattern (GraphSAGE) - no cost anomaly on its own"
    if xgb_lab == "HIGH":
        return "HIGH", "cost anomaly (XGBoost)"
    if xgb_lab == "MEDIUM":
        return "MEDIUM", "cost anomaly (XGBoost, below auto-flag threshold)"
    return "LOW", "neither model flagged this claim"


def label_badge(label: str) -> str:
    return {"HIGH": "\U0001F534 HIGH RISK – auto-flag for fraud review",
            "MEDIUM": "\U0001F7E1 MEDIUM RISK – route to manual audit",
            "LOW": "\U0001F7E2 LOW RISK – clear"}[label]


def log_claim(conn, raw, source, username, score, label, gs_score=None, flagged_by=None):
    cur = conn.execute("""
        INSERT INTO claims_log (
            submitted_at, submitted_by, source, patient_id, provider_id,
            primary_condition, encounter_type, gender, county, service_date,
            admission_datetime, discharge_datetime,
            age, length_of_stay_hours, base_cost_kes, claim_amount_kes,
            payer_coverage_kes, claim_to_base_ratio, fraud_score, risk_label,
            graphsage_score, flagged_by, review_status
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        dt.datetime.now().isoformat(timespec="seconds"), username, source,
        raw.get("PATIENT_ID"), str(raw["PROVIDER_ID"]), raw["PRIMARY_CONDITION"],
        raw["ENCOUNTER_TYPE"], raw["GENDER"], raw["PATIENT_COUNTY"], str(raw["SERVICE_DATE"]),
        raw.get("ADMISSION_DATETIME"), raw.get("DISCHARGE_DATETIME"),
        raw["AGE"], raw["LENGTH_OF_STAY_HOURS"],
        raw["BASE_COST_KES"], raw["CLAIM_AMOUNT_KES"], raw["PAYER_COVERAGE_KES"],
        raw["CLAIM_TO_BASE_RATIO"], score, label, gs_score, flagged_by, "pending",
    ))
    conn.commit()
    return cur.lastrowid


def save_claim_items(conn, claim_id, lines):
    for ln in lines:
        conn.execute(
            "INSERT INTO claim_items (claim_id, category, item, qty, tariff_price, billed_price) VALUES (?,?,?,?,?,?)",
            (claim_id, ln["Category"], ln["Item"], ln["Qty"], ln["Tariff price"], ln["Billed price"]),
        )
    conn.commit()


def log_event(conn, claim_id, actor, event_type, detail):
    conn.execute(
        "INSERT INTO claim_events (claim_id, created_at, actor, event_type, detail) VALUES (?,?,?,?,?)",
        (claim_id, dt.datetime.now().isoformat(timespec="seconds"), actor, event_type, detail),
    )
    conn.commit()


def build_claim_report(cid, fields: dict, scores: dict, flags: list, shap_lines: list, events: list, items=None) -> str:
    """Plain-text case file for one claim - for an auditor to attach to a
    referral. Built only from data already shown on screen."""
    out = [f"CLAIM REPORT - Claim #{cid}",
           f"Generated {dt.datetime.now().isoformat(timespec='seconds')}", "", "CLAIM DETAILS"]
    out += [f"  {k}: {v}" for k, v in fields.items()]
    if items:
        out += ["", "ITEMISED BILL (item | qty | tariff | billed)"]
        out += [f"  {c} | {i} | {q:g} | {t:,.0f} | {b:,.0f}" for c, i, q, t, b in items]
    out += ["", "MODEL SCORES"]
    out += [f"  {k}: {v}" for k, v in scores.items()]
    out += ["", "ANOMALY FLAGS"] + ([f"  - {f}" for f in flags] or ["  none"])
    out += ["", "TOP SHAP DRIVERS (XGBoost)"] + ([f"  - {l}" for l in shap_lines] or ["  not available"])
    out += ["", "REVIEW TIMELINE AND NOTES"]
    out += [f"  {t}  [{typ}] {actor}: {detail}" for t, actor, typ, detail in events] or ["  no review activity yet"]
    out += ["", "Model output is decision support, not a finding of fraud."]
    return "\n".join(out)


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
        scheme = st.selectbox("Insurance scheme", list(cref.SCHEMES),
                              help="Determines how much of the tariff the payer covers on claims.")
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
                "registered_by, registered_at, scheme) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (patient_id, first_name.strip(), last_name.strip(), gender, dob.isoformat(), age, county,
                 st.session_state.username, dt.datetime.now().isoformat(timespec="seconds"), scheme),
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
def encounter_time_inputs(encounter_type, key, start_default=None, end_default=None, disabled=False):
    """Visit start/end (admission/discharge for inpatients). Length of stay is
    always CALCULATED from these, never typed. If no end is supplied, the end
    defaults to start + the typical stay for this encounter type (median in
    the training data), which staff can edit."""
    is_inp = encounter_type == "inpatient"
    s_lbl, e_lbl = ("Admission", "Discharge") if is_inp else ("Visit start", "Visit end")
    start0 = start_default or dt.datetime.combine(dt.date.today(), dt.time(9, 0))
    end0 = end_default or (start0 + dt.timedelta(hours=cref.TYPICAL_LOS_HOURS.get(encounter_type, 12)))
    a1, a2 = st.columns(2)
    sd = a1.date_input(f"{s_lbl} date", value=start0.date(), key=f"sd_{key}_{encounter_type}", disabled=disabled)
    stm = a2.time_input(f"{s_lbl} time", value=start0.time(), key=f"st_{key}_{encounter_type}", disabled=disabled)
    b1, b2 = st.columns(2)
    ed = b1.date_input(f"{e_lbl} date", value=end0.date(), key=f"ed_{key}_{encounter_type}", disabled=disabled)
    etm = b2.time_input(f"{e_lbl} time", value=end0.time(), key=f"et_{key}_{encounter_type}", disabled=disabled)
    start_dt = dt.datetime.combine(sd, stm)
    end_dt = dt.datetime.combine(ed, etm)
    los = max((end_dt - start_dt).total_seconds() / 3600, 0.0)
    st.metric("Length of stay (hours) - calculated", round(los, 1))
    if end_dt < start_dt:
        st.error(f"{e_lbl} can't be before {s_lbl.lower()}.")
    return start_dt, end_dt, los


def record_encounter_page(conn, artifacts):
    st.header("Record encounter")
    st.caption(
        "Clinical record of a visit: diagnosis, admission/visit times, and the medication, tests and "
        "procedures ordered. The provider's billing team then picks this encounter on Submit claim, "
        "prices it, and submits the claim - so the claim is built from what clinically happened."
    )
    patients_df = pd.read_sql_query(
        "SELECT patient_id, first_name, last_name, gender, county FROM patients ORDER BY registered_at DESC LIMIT 300",
        conn,
    )
    if patients_df.empty:
        st.info("Register a patient first (Register patient page), then record their encounter here.")
        return
    pid_list = patients_df["patient_id"].tolist()
    patient_id = st.selectbox(
        "Patient", pid_list,
        format_func=lambda x: f"{x} - {patients_df.loc[patients_df.patient_id == x, 'first_name'].iloc[0]} "
                              f"{patients_df.loc[patients_df.patient_id == x, 'last_name'].iloc[0]}",
    )
    org_names = artifacts.get("organization_names", {})
    org_lookup = artifacts["provider_org_lookup"]
    c1, c2 = st.columns(2)
    with c1:
        encounter_type = st.selectbox("Encounter type", artifacts["encounter_types"], key="rec_etype")
        provider_id = st.selectbox(
            "Facility / provider", artifacts["top_providers"],
            format_func=lambda pid: f"{org_names.get(org_lookup.get(pid, ''), 'Facility ' + org_lookup.get(pid, 'unknown')[:8])}"
                                    f"  ·  Provider {pid[:8]}",
            key="rec_provider",
        )
        condition = st.selectbox("Diagnosis (primary condition)", artifacts["conditions"], key="rec_cond")
        ward = None
        if encounter_type == "inpatient":
            ward = st.selectbox("Ward", list(cref.WARD_RATES), key="rec_ward")
    with c2:
        start_dt, end_dt, _los = encounter_time_inputs(encounter_type, "rec")

    st.subheader("Medication, tests and procedures ordered")
    st.caption("Quantities only - prices are added later when the claim is billed.")
    bundle_key = f"rec_bundle_{encounter_type}"
    if bundle_key not in st.session_state:
        st.session_state[bundle_key] = pd.DataFrame(
            [{"Item": i, "Qty": q} for i, q in cref.DEFAULT_BUNDLES.get(encounter_type, [])],
            columns=["Item", "Qty"],
        )
    items = st.data_editor(
        st.session_state[bundle_key], num_rows="dynamic", use_container_width=True,
        key=f"rec_editor_{encounter_type}",
        column_config={
            "Item": st.column_config.SelectboxColumn("Item", options=cref.ITEM_NAMES, required=True, width="large"),
            "Qty": st.column_config.NumberColumn("Qty", min_value=1, step=1, default=1),
        },
    )
    notes = st.text_area("Clinical notes (optional)")
    if st.button("Save encounter", type="primary"):
        valid = [(r["Item"], float(r["Qty"]) if pd.notna(r["Qty"]) else 1.0)
                 for _, r in items.iterrows() if isinstance(r.get("Item"), str) and r["Item"] in cref.CATALOG_BY_ITEM]
        if end_dt < start_dt:
            st.error("End can't be before start.")
        elif not valid:
            st.error("Add at least one service, medication or test.")
        else:
            cur = conn.execute(
                "INSERT INTO encounters (patient_id, provider_id, recorded_by, recorded_at, encounter_type, "
                "primary_condition, start_datetime, end_datetime, ward, notes, status) VALUES (?,?,?,?,?,?,?,?,?,?,'open')",
                (patient_id, provider_id, st.session_state.username, dt.datetime.now().isoformat(timespec="seconds"),
                 encounter_type, condition, start_dt.isoformat(), end_dt.isoformat(), ward, notes.strip() or None),
            )
            eid = cur.lastrowid
            for item, qty in valid:
                conn.execute("INSERT INTO encounter_items (encounter_id, item, qty) VALUES (?,?,?)", (eid, item, qty))
            conn.commit()
            st.success(f"Encounter E-{eid} saved. It now appears on Submit claim (provider) as an unbilled encounter.")

    st.divider()
    st.subheader("Recent encounters")
    rec = pd.read_sql_query(
        "SELECT 'E-' || id AS Encounter, patient_id AS Patient, encounter_type AS Type, primary_condition AS Diagnosis, "
        "start_datetime AS Start, status AS Status FROM encounters ORDER BY id DESC LIMIT 20",
        conn,
    )
    if rec.empty:
        st.caption("No encounters recorded yet.")
    else:
        st.dataframe(rec, use_container_width=True, hide_index=True)


def render_patient_history(conn, patient_id, condition, scheme):
    """Context panel shown on the claim form: who the patient is, their
    scheme, and what the system already knows about their recent claims."""
    st.subheader("Patient history")
    if patient_id is None:
        st.caption("Walk-in - no patient record linked, so there is no history to show. "
                   f"Coverage will be calculated under the default scheme ({cref.DEFAULT_SCHEME}).")
        return
    p = conn.execute(
        "SELECT first_name, last_name, gender, dob, age, county FROM patients WHERE patient_id = ?",
        (patient_id,),
    ).fetchone()
    if p:
        age_now = calculate_age(dt.date.fromisoformat(p[3])) if p[3] else p[4]
        st.write(f"**{p[0]} {p[1]}**  ·  {p[2]}  ·  age {age_now}  ·  {p[5]} county  ·  scheme: **{scheme}**")
    hist = pd.read_sql_query(
        "SELECT id, submitted_at, service_date, provider_id, primary_condition, encounter_type, "
        "claim_amount_kes, risk_label FROM claims_log WHERE patient_id = ? ORDER BY id DESC",
        conn, params=(patient_id,),
    )
    cutoff = dt.datetime.now() - dt.timedelta(days=90)
    if hist.empty:
        recent = hist
    else:
        recent = hist[pd.to_datetime(hist["submitted_at"]) >= cutoff]
    repeats = int((recent["primary_condition"] == condition).sum()) if not recent.empty else 0
    providers_90 = recent["provider_id"].nunique() if not recent.empty else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Claims on file", len(hist))
    c2.metric("Last 90 days", len(recent))
    c3.metric("Same diagnosis (90d)", repeats)
    c4.metric("Providers seen (90d)", providers_90)
    if repeats >= 2:
        st.warning(f"This patient already has {repeats} claim(s) for this diagnosis in the last 90 days.")
    if providers_90 >= 3:
        st.warning(f"This patient has visited {providers_90} different providers in the last 90 days.")
    if not hist.empty:
        show = hist.head(5)[["service_date", "primary_condition", "encounter_type", "claim_amount_kes", "risk_label"]]
        show = show.rename(columns={"service_date": "Date", "primary_condition": "Diagnosis",
                                    "encounter_type": "Type", "claim_amount_kes": "Amount (KES)",
                                    "risk_label": "Risk"})
        st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.caption("No earlier claims for this patient in this system.")
    nb = load_graph_neighborhood()
    if nb is not None and str(patient_id) in nb["patient_top_providers"]:
        n_prov = len(nb["patient_top_providers"][str(patient_id)])
        st.caption(f"Historical records (training data): this patient has claims with {n_prov}+ provider(s).")


def render_bed_panel(conn, provider_id, admission_dt):
    """Inpatient only: illustrative facility bed capacity/occupancy on the
    admission date, plus inpatients already logged in this app overlapping it."""
    st.subheader("Facility bed availability")
    capacity, base_pct = cref.facility_beds(provider_id)
    ref = admission_dt or dt.datetime.now()
    occupied_base = round(capacity * base_pct / 100)
    overlap = conn.execute(
        "SELECT COUNT(*) FROM claims_log WHERE provider_id = ? AND admission_datetime IS NOT NULL "
        "AND encounter_type = 'inpatient' AND admission_datetime <= ? AND discharge_datetime >= ?",
        (provider_id, ref.isoformat(), ref.isoformat()),
    ).fetchone()[0]
    occupied = min(capacity, occupied_base + overlap)
    available = capacity - occupied
    b1, b2, b3 = st.columns(3)
    b1.metric("Bed capacity", capacity)
    b2.metric("Estimated occupied", occupied)
    b3.metric("Available", available)
    if available <= 0:
        st.error("No beds available at this facility on the admission date - an inpatient claim here needs scrutiny.")
    elif available <= max(1, capacity // 20):
        st.warning("Facility is close to full on the admission date.")
    st.caption("Illustrative demo data (bed counts are not in the source dataset).")


def submit_claim_page(conn, model, artifacts, gs_artifacts=None):
    st.header("Submit a claim")
    fd = artifacts["form_defaults"]
    org_names = artifacts.get("organization_names", {})
    org_lookup = artifacts["provider_org_lookup"]

    def _provider_label(pid: str) -> str:
        org_id = org_lookup.get(pid, "unknown")
        name = org_names.get(org_id, f"Facility {org_id[:8]}")
        return f"{name}  ·  Provider {pid[:8]}"

    # ---- start from a clinician-recorded encounter (preferred) or manually ----
    enc = None
    mode = st.radio("Start this claim from", ["A recorded encounter (recommended)", "Enter details manually"],
                    horizontal=True)
    if mode.startswith("A recorded"):
        enc_rows = conn.execute(
            "SELECT e.id, e.patient_id, p.first_name, p.last_name, e.provider_id, e.encounter_type, "
            "e.primary_condition, e.start_datetime, e.end_datetime, e.ward FROM encounters e "
            "LEFT JOIN patients p ON p.patient_id = e.patient_id WHERE e.status = 'open' ORDER BY e.id DESC LIMIT 200"
        ).fetchall()
        if not enc_rows:
            st.info("No unbilled encounters yet - a nurse records them on Record encounter. "
                    "Switch to 'Enter details manually' to bill without one.")
        else:
            enc_labels = {
                r[0]: f"E-{r[0]} · {r[2] or ''} {r[3] or ''} ({r[1]}) · {r[5]} · {r[6][:38]} · {r[7][:10]}"
                for r in enc_rows
            }
            chosen = st.selectbox("Recorded encounter (not yet billed)", list(enc_labels), format_func=enc_labels.get)
            r = next(x for x in enc_rows if x[0] == chosen)
            enc = {"id": r[0], "patient_id": r[1], "provider_id": r[4], "encounter_type": r[5],
                   "condition": r[6], "start": dt.datetime.fromisoformat(r[7]),
                   "end": dt.datetime.fromisoformat(r[8]), "ward": r[9]}
            st.caption("Patient, facility, diagnosis, times and services come from the clinical record "
                       "(locked). You add the billing: prices, then submit.")
    ek = enc["id"] if enc else "manual"
    lock = enc is not None

    patients_df = pd.read_sql_query(
        "SELECT patient_id, first_name, last_name, dob, age FROM patients "
        "ORDER BY registered_at DESC LIMIT 300",
        conn,
    )
    if enc is not None:
        patient_id = enc["patient_id"]
    elif patients_df.empty:
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

    # Age is calculated from the patient's date of birth; gender and county
    # default to the patient's record.
    prow = None
    if patient_id is not None:
        prow = conn.execute(
            "SELECT first_name, last_name, gender, dob, age, county, scheme FROM patients WHERE patient_id = ?",
            (patient_id,),
        ).fetchone()
    linked_age = None
    if prow is not None:
        if prow[3]:
            linked_age = calculate_age(dt.date.fromisoformat(prow[3]))
        elif prow[4] is not None:
            linked_age = int(prow[4])
    if enc is not None:
        st.write(f"**Patient:** {prow[0]} {prow[1]} ({patient_id})" if prow else f"**Patient:** {patient_id}")

    types = artifacts["encounter_types"]
    encounter_type = st.selectbox(
        "Encounter type", types,
        index=types.index(enc["encounter_type"]) if enc and enc["encounter_type"] in types else 0,
        disabled=lock, key=f"etype_{ek}",
    )
    is_inpatient = encounter_type == "inpatient"

    def _idx(options, value):
        return options.index(value) if value in options else 0

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            providers = artifacts["top_providers"]
            provider_id = st.selectbox(
                "Provider", providers, format_func=_provider_label,
                index=_idx(providers, enc["provider_id"]) if enc else 0,
                disabled=lock, key=f"prov_{ek}",
            )
            org_id = org_lookup.get(provider_id, "unknown")
            org_name = org_names.get(org_id, "unknown facility")
            st.caption(f"Facility: {org_name}")
            conds = artifacts["conditions"]
            condition = st.selectbox("Primary condition", conds,
                                     index=_idx(conds, enc["condition"]) if enc else 0,
                                     disabled=lock, key=f"cond_{ek}")
            counties = artifacts["counties"]
            county = st.selectbox("Patient county", counties,
                                  index=_idx(counties, prow[5]) if prow else 0, key=f"cty_{ek}_{patient_id}")
            genders = artifacts["genders"]
            gender = st.selectbox("Gender", genders,
                                  index=_idx(genders, prow[2]) if prow else 0, key=f"gen_{ek}_{patient_id}")
        with col2:
            admission_dt, discharge_dt, length_of_stay = encounter_time_inputs(
                encounter_type, ek,
                start_default=enc["start"] if enc else None,
                end_default=enc["end"] if enc else None,
                disabled=lock,
            )
            service_date = admission_dt.date()
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
    # ---------------- context panels ----------------
    scheme = prow[6] if prow is not None and prow[6] else cref.DEFAULT_SCHEME
    st.divider()
    render_patient_history(conn, patient_id, condition, scheme)
    if is_inpatient:
        render_bed_panel(conn, provider_id, admission_dt)

    # ---------------- itemised bill ----------------
    st.divider()
    st.subheader("Itemised bill")
    st.caption(
        "Add the services, medication and tests billed. Base cost is the sum of the reference tariff prices; "
        "the claim amount is the sum of what is actually billed (edit a billed unit price to bill above/below "
        "tariff). Payer coverage is calculated from the patient's scheme, not typed in."
    )
    if st.session_state.get("bill_bundle_for") != (encounter_type, ek):
        st.session_state.bill_bundle_for = (encounter_type, ek)
        if enc is not None:
            seed_items = [(i, q) for i, q in conn.execute(
                "SELECT item, qty FROM encounter_items WHERE encounter_id = ? ORDER BY id", (enc["id"],)).fetchall()]
        else:
            seed_items = cref.DEFAULT_BUNDLES.get(encounter_type, [])
        st.session_state.bill_df = pd.DataFrame(
            [{"Item": i, "Qty": q, "Billed unit price (KES)": cref.tariff_price(i)} for i, q in seed_items],
            columns=["Item", "Qty", "Billed unit price (KES)"],
        )
    lines = []
    if is_inpatient:
        wc1, wc2 = st.columns(2)
        ward = wc1.selectbox("Ward type", list(cref.WARD_RATES),
                             index=_idx(list(cref.WARD_RATES), enc["ward"]) if enc else 0, key=f"ward_{ek}")
        calc_days = cref.bed_days(length_of_stay)
        days_billed = wc2.number_input(
            f"Bed-days billed (stay of {length_of_stay:.1f} h = {calc_days} day(s))",
            min_value=1, value=calc_days, key=f"bed_days_{calc_days}_{ek}",
        )
        rate = float(cref.WARD_RATES[ward])
        lines.append({"Category": "Accommodation", "Item": f"{ward} bed-day", "Qty": float(days_billed),
                      "Tariff price": rate, "Billed price": rate})
        if days_billed > calc_days:
            st.warning(f"{days_billed} bed-days billed but the recorded stay only covers {calc_days}.")

    edited = st.data_editor(
        st.session_state.bill_df, num_rows="dynamic", use_container_width=True,
        key=f"bill_editor_{encounter_type}_{ek}",
        column_config={
            "Item": st.column_config.SelectboxColumn("Item", options=cref.ITEM_NAMES, required=True, width="large"),
            "Qty": st.column_config.NumberColumn("Qty", min_value=1, step=1, default=1),
            "Billed unit price (KES)": st.column_config.NumberColumn("Billed unit price (KES)", min_value=0.0, format="%.0f"),
        },
    )
    over_tariff = []
    for _, r in edited.iterrows():
        item = r.get("Item")
        if not isinstance(item, str) or item not in cref.CATALOG_BY_ITEM:
            continue
        qty = float(r["Qty"]) if pd.notna(r["Qty"]) else 1.0
        tp = cref.tariff_price(item)
        bp = float(r["Billed unit price (KES)"]) if pd.notna(r["Billed unit price (KES)"]) else tp
        lines.append({"Category": cref.item_category(item), "Item": item, "Qty": qty,
                      "Tariff price": tp, "Billed price": bp})
        if tp and bp > tp * 1.5:
            over_tariff.append(f"{item} (billed {bp:,.0f} vs tariff {tp:,.0f})")
    if over_tariff:
        st.warning("Billed well above reference price: " + "; ".join(over_tariff))

    base_cost = round(sum(l["Qty"] * l["Tariff price"] for l in lines), 2)
    claim_amount = round(sum(l["Qty"] * l["Billed price"] for l in lines), 2)
    payer_coverage = cref.payer_coverage(encounter_type, scheme, base_cost, claim_amount)
    claim_to_base_ratio = round(claim_amount / base_cost, 3) if base_cost else 1.0
    if lines:
        summary = pd.DataFrame(lines)
        summary["Line total (KES)"] = summary["Qty"] * summary["Billed price"]
        summary["Price basis"] = summary["Item"].map(lambda i: cref.price_basis(i) if i in cref.CATALOG_BY_ITEM else "Ward tariff (SHA inpatient/ICU rates)")
        st.dataframe(summary.rename(columns={"Tariff price": "Tariff (KES)", "Billed price": "Billed (KES)"}),
                     use_container_width=True, hide_index=True)
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Base cost (tariff)", f"{base_cost:,.0f}")
    t2.metric("Claim amount", f"{claim_amount:,.0f}")
    t3.metric("Payer coverage", f"{payer_coverage:,.0f}", help=f"Calculated for scheme: {scheme}")
    t4.metric("Patient balance", f"{claim_amount - payer_coverage:,.0f}")
    t5.metric("Claim-to-base ratio", claim_to_base_ratio)
    st.caption("Reference prices come from published SHA/SHIF rates, public-hospital service charters, KNH "
               "procurement costs and market ranges (see 'Price basis'); items marked Estimate have no "
               "published source. Bed counts and scheme rules are illustrative - see claim_reference.py.")

    submitted = st.button("Submit claim", type="primary", disabled=not lines,
                          help="Submits this claim and scores it immediately; it then appears on the claims dashboard for audit review.")

    if submitted and discharge_dt < admission_dt:
        st.error("End date/time can't be before start date/time.")
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
        score, xgb_only_label = score_row(feat, model, artifacts)
        gs_score = graphsage_ring_score(patient_id, provider_id, gs_artifacts)
        label, flagged_by = hybrid_label(score, gs_score, artifacts, gs_artifacts)
        new_claim_id = log_claim(conn, raw, "manual", st.session_state.username, score, label, gs_score, flagged_by)
        save_claim_items(conn, new_claim_id, lines)
        if enc is not None:
            conn.execute("UPDATE encounters SET status = 'billed', claim_id = ? WHERE id = ?", (new_claim_id, enc["id"]))
            conn.commit()

        st.divider()
        st.subheader(f"Claim #{new_claim_id} submitted - result")
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.metric("Cost-anomaly score (XGBoost)", f"{score:.3f}")
        with c2:
            if gs_score is not None:
                st.metric("Relational score (GraphSAGE)", f"{gs_score:.3f}")
            else:
                st.metric("Relational score (GraphSAGE)", "n/a")
        with c3:
            if label == "HIGH":
                st.error(label_badge(label))
            elif label == "MEDIUM":
                st.warning(label_badge(label))
            else:
                st.success(label_badge(label))
        st.caption(f"Flagged by: {flagged_by}")
        if gs_score is None:
            st.caption(
                "No GraphSAGE relational score: this patient and/or provider has no claim "
                "history in the training graph yet, so there's nothing to measure a repeating "
                "pattern against - the result above is from XGBoost's cost-anomaly read alone."
            )
        st.caption(
            f"XGBoost thresholds: score < {artifacts['t_low']:.3f} = low risk, "
            f"{artifacts['t_low']:.3f}–{artifacts['t_high']:.3f} = medium (manual review), "
            f">= {artifacts['t_high']:.3f} = high (auto-flag)."
            + (f"  GraphSAGE auto-flags at >= {gs_artifacts['t_graphsage']:.2f} "
               f"(tuned on a held-out validation split, see tune_hybrid_threshold.py)."
               if gs_artifacts else "")
        )


# ============================================================
# CSV UPLOAD (BATCH SCORING)
# ============================================================
REQUIRED_CSV_COLUMNS = [
    "PROVIDER_ID", "PRIMARY_CONDITION", "PATIENT_COUNTY", "ENCOUNTER_TYPE",
    "GENDER", "SERVICE_DATE", "AGE", "LENGTH_OF_STAY_HOURS",
    "BASE_COST_KES", "CLAIM_AMOUNT_KES", "PAYER_COVERAGE_KES", "CLAIM_TO_BASE_RATIO",
]


def upload_csv_page(conn, model, artifacts, gs_artifacts=None):
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
    scores, labels, gs_scores, flagged_bys = [], [], [], []
    for _, row in df.iterrows():
        raw = row[REQUIRED_CSV_COLUMNS].to_dict()
        raw["PATIENT_ID"] = row.get("PATIENT_ID") if "PATIENT_ID" in df.columns and pd.notna(row.get("PATIENT_ID")) else None
        feat = engineer_row(raw, artifacts, conn)
        score, _ = score_row(feat, model, artifacts)
        gs_score = graphsage_ring_score(raw["PATIENT_ID"], raw["PROVIDER_ID"], gs_artifacts)
        label, flagged_by = hybrid_label(score, gs_score, artifacts, gs_artifacts)
        log_claim(conn, raw, "csv", st.session_state.username, score, label, gs_score, flagged_by)
        scores.append(score)
        labels.append(label)
        gs_scores.append(gs_score)
        flagged_bys.append(flagged_by)

    df["FRAUD_SCORE"] = scores
    df["GRAPHSAGE_SCORE"] = gs_scores
    df["RISK_LABEL"] = labels
    df["FLAGGED_BY"] = flagged_bys

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

    # A provider only ever sees the claims THEY submitted (not other
    # facilities' claims); auditors see everything. Review decisions are
    # read straight from the shared database, so an auditor's approve/reject/
    # escalate shows up here on the provider's next page load.
    is_provider = st.session_state.get("role") == "provider"
    where = "WHERE submitted_by = ? " if is_provider else ""
    params = (st.session_state.get("username"),) if is_provider else ()
    df = pd.read_sql_query(
        "SELECT id, submitted_at, patient_id, provider_id, primary_condition, "
        "claim_amount_kes, fraud_score, risk_label, review_status, "
        "graphsage_score, flagged_by FROM claims_log " + where +
        "ORDER BY id DESC LIMIT 1000",
        conn, params=params,
    )
    if df.empty:
        st.info("No claims submitted yet." if is_provider else "No claims scored yet.")
        return

    if is_provider:
        st.caption("Showing the claims you submitted. Status updates when an auditor reviews a claim.")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("My claims", len(df))
        c2.metric("Pending review", int((df["review_status"] == "pending").sum()))
        c3.metric("Approved", int((df["review_status"] == "approved").sum()))
        c4.metric("Rejected", int((df["review_status"] == "rejected").sum()))
        c5.metric("Escalated", int((df["review_status"] == "escalated").sum()))
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total claims", len(df))
        c2.metric("High risk", int((df["risk_label"] == "HIGH").sum()))
        c3.metric("Pending review", int((df["review_status"] == "pending").sum()))

    search_col, risk_col, status_col, flag_col = st.columns([2, 1, 1, 1.3])
    search = search_col.text_input("Search claims (claim ID, patient ID, provider ID)")
    risk_filter = risk_col.selectbox("Risk level", ["All", "HIGH", "MEDIUM", "LOW"])
    status_filter = status_col.selectbox("Status", ["All", "pending", "approved", "rejected", "escalated"])
    flag_filter = flag_col.selectbox(
        "Flagged by",
        ["All", "XGBoost only", "GraphSAGE only", "Both models", "Neither"],
        help="Which model(s) raised this claim's risk label - surfaces the hybrid "
             "architecture's actual contribution, not just the final verdict.",
    )

    org_names = artifacts.get("organization_names", {})
    org_lookup = artifacts["provider_org_lookup"]
    df["facility"] = df["provider_id"].map(lambda pid: org_names.get(org_lookup.get(pid, ""), "unknown facility"))

    # "flagged_by" is a free-text reason string set by hybrid_label() (e.g.
    # "cost anomaly (XGBoost)", "relational pattern (GraphSAGE) - no cost
    # anomaly on its own", "both models ..."). Bucket it into the four
    # mutually-exclusive categories the filter above offers. Claims logged
    # before the hybrid wiring (flagged_by IS NULL) fall into "Neither" here
    # rather than being silently dropped from every bucketed filter.
    def _flag_bucket(reason):
        if not isinstance(reason, str) or not reason:
            return "Neither"
        if "both models" in reason:
            return "Both models"
        if "GraphSAGE" in reason:
            return "GraphSAGE only"
        if "XGBoost" in reason:
            return "XGBoost only"
        return "Neither"

    df["flag_bucket"] = df["flagged_by"].map(_flag_bucket)

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
    if flag_filter != "All":
        filtered = filtered[filtered["flag_bucket"] == flag_filter]

    st.caption(f"{len(filtered)} claim(s). Click a row to open its Claim Detail screen.")

    display_df = filtered[[
        "id", "patient_id", "facility", "primary_condition", "claim_amount_kes",
        "fraud_score", "graphsage_score", "risk_label", "flagged_by", "review_status",
    ]].rename(columns={
        "id": "Claim ID", "patient_id": "Patient", "facility": "Provider",
        "primary_condition": "Diagnosis", "claim_amount_kes": "Amount (KES)",
        "fraud_score": "XGBoost score", "graphsage_score": "GraphSAGE score",
        "risk_label": "Risk label", "flagged_by": "Flagged by", "review_status": "Status",
    })

    event = st.dataframe(
        display_df, use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row",
    )
    st.download_button("Export filtered claims (CSV)", display_df.to_csv(index=False),
                       file_name="claims_export.csv", mime="text/csv")
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
        "graphsage_score, flagged_by, "
        "review_status, reviewed_by, reviewed_at, override_reason FROM claims_log WHERE id = ?",
        (claim_id,),
    ).fetchone()
    if row is None:
        st.error("Claim not found.")
        return

    (cid, submitted_at, submitted_by, source, patient_id, provider_id, condition, encounter_type,
     gender, county, service_date, admission_datetime, discharge_datetime, age, los, base_cost,
     claim_amount, payer_coverage,
     ratio, score, label, gs_score, flagged_by,
     review_status, reviewed_by, reviewed_at, override_reason) = row

    if role == "provider" and submitted_by != st.session_state.get("username"):
        st.error("You can only open claims you submitted.")
        return

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
            st.write(f"**{'Admission' if encounter_type == 'inpatient' else 'Visit start'}:** {admission_datetime}")
            st.write(f"**{'Discharge' if encounter_type == 'inpatient' else 'Visit end'}:** {discharge_datetime}")
        st.write(f"**Length of stay:** {los:.1f} hours")
        st.write(f"**Base cost (SHA tariff):** KES {base_cost:,.0f}")
        st.write(f"**Claim amount:** KES {claim_amount:,.0f}")
        st.write(f"**Payer coverage:** KES {payer_coverage:,.0f}")
        st.write(f"**Claim-to-base ratio:** {ratio:.2f}")
        st.caption(f"Submitted by {submitted_by} ({source}) on {submitted_at}")
        item_rows = conn.execute(
            "SELECT category, item, qty, tariff_price, billed_price FROM claim_items WHERE claim_id = ? ORDER BY id",
            (cid,),
        ).fetchall()
        if item_rows:
            st.subheader("Itemised bill")
            idf = pd.DataFrame(item_rows, columns=["Category", "Item", "Qty", "Tariff (KES)", "Billed (KES)"])
            idf["Line total (KES)"] = idf["Qty"] * idf["Billed (KES)"]
            st.dataframe(idf, use_container_width=True, hide_index=True)
            over = idf[idf["Billed (KES)"] > idf["Tariff (KES)"] * 1.5]
            if not over.empty:
                st.warning("Billed well above tariff: " + "; ".join(over["Item"].tolist()))

    with col_right:
        st.subheader("Fraud risk score")
        gs_artifacts = load_graphsage_artifacts()
        # Claims logged before GraphSAGE was wired in have graphsage_score
        # = NULL on file; recompute it live for display here (read-only -
        # doesn't rewrite the historical log row) rather than showing a
        # stale "not available" for claims that could be scored now.
        if gs_score is None and gs_artifacts is not None:
            gs_score = graphsage_ring_score(patient_id, provider_id, gs_artifacts)
            if flagged_by is None:
                _, flagged_by = hybrid_label(score, gs_score, artifacts, gs_artifacts)

        sc1, sc2, sc3 = st.columns([1, 1, 1.5])
        with sc1:
            st.metric("Cost-anomaly (XGBoost)", f"{score:.3f}")
        with sc2:
            st.metric("Relational (GraphSAGE)", f"{gs_score:.3f}" if gs_score is not None else "n/a")
        with sc3:
            if label == "HIGH":
                st.error(label_badge(label))
            elif label == "MEDIUM":
                st.warning(label_badge(label))
            else:
                st.success(label_badge(label))
        if flagged_by:
            st.caption(f"Flagged by: {flagged_by}")
        if gs_score is None:
            st.caption(
                "GraphSAGE has no relational score for this claim - the linked patient and/or "
                "provider has no claim history in the training graph, so there's no repeating "
                "pattern to measure yet. Verdict above reflects XGBoost's cost-anomaly read alone."
            )

        # Recompute the engineered feature row once here - needed for the
        # anomaly-flag badges below AND for SHAP, so it's done unconditionally
        # (previously this only ran inside the SHAP branch, meaning a claim's
        # geographic/clinical anomaly flags were only ever visible when they
        # happened to rank in the SHAP top-3 - easy to miss for an auditor
        # scanning quickly).
        raw = {
            "PROVIDER_ID": provider_id, "PRIMARY_CONDITION": condition, "PATIENT_COUNTY": county,
            "ENCOUNTER_TYPE": encounter_type, "GENDER": gender, "SERVICE_DATE": service_date,
            "AGE": age, "LENGTH_OF_STAY_HOURS": los, "BASE_COST_KES": base_cost,
            "CLAIM_AMOUNT_KES": claim_amount, "PAYER_COVERAGE_KES": payer_coverage,
            "CLAIM_TO_BASE_RATIO": ratio,
        }
        feat = None
        feat_error = None
        shap_lines = []
        report_flags = []
        try:
            feat = engineer_row(raw, artifacts, conn)
        except Exception as e:
            feat_error = e

        st.subheader("Relational view (GraphSAGE)")
        nb = load_graph_neighborhood()
        if nb is None or gs_artifacts is None or not patient_id:
            st.caption("Relational view unavailable - needs a linked patient and the exported "
                       "graph_neighborhood.json (run export_graph_neighborhood.py).")
        elif str(patient_id) not in nb["patient_top_providers"] and str(provider_id) not in nb["provider_top_patients"]:
            st.caption("This patient and provider have no history in the graph yet, so there is "
                       "no neighbourhood to draw.")
        else:
            dot, n_pat, n_prov, n_edges = build_neighborhood_dot(
                patient_id, provider_id, nb, gs_artifacts, org_names, org_lookup)
            st.graphviz_chart(dot, use_container_width=True)
            st.caption(
                f"{n_pat} patients, {n_prov} providers, {n_edges} links (labels = claims between the pair; "
                "line width scales with claim count). Red links are pairs GraphSAGE itself rates at or above its "
                "auto-flag threshold - a dense red mesh of the same few patients across several providers is the "
                "collusion-ring signature. Showing top partners only."
            )

        st.subheader("Anomaly flags")
        if feat is None:
            st.caption(f"Anomaly flags unavailable: {feat_error}")
        else:
            geo_flag = bool(feat.get("GEOGRAPHIC_MISMATCH"))
            clinical_flag = bool(feat.get("CLINICAL_INCONSISTENCY"))
            if geo_flag:
                report_flags.append("Geographic mismatch - facility outside patient's home county")
            if clinical_flag:
                report_flags.append("Clinical inconsistency - diagnosis does not fit recorded gender")
            if not geo_flag and not clinical_flag:
                st.caption("No geographic or clinical inconsistency flags on this claim.")
            else:
                fcol1, fcol2 = st.columns(2)
                if geo_flag:
                    fcol1.warning(
                        "⚠️ **Geographic mismatch** - this facility is outside "
                        "the patient's home county.",
                        icon="\U0001F4CD",
                    )
                if clinical_flag:
                    fcol2.warning(
                        "⚠️ **Clinical inconsistency** - the diagnosis doesn't "
                        "fit this patient's recorded gender.",
                        icon="\U0001FA7A",
                    )
                st.caption(
                    "These are always shown when present - independent of whether "
                    "SHAP ranks them in the top 3 contributing features below."
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
        elif feat is None:
            st.caption(f"SHAP explanation unavailable: {feat_error}")
        else:
            try:
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
                    shap_lines.append(f"{fname_label} ({feat[fname]:.2f}) {direction} the fraud score ({val:+.3f})")
            except Exception as e:
                st.caption(f"SHAP explanation unavailable: {e}")

    st.divider()
    st.subheader("Auditor decision")
    if review_status != "pending":
        note = f"Already reviewed: **{review_status}** by {reviewed_by} on {reviewed_at}"
        if override_reason and role == "auditor":
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
            log_event(conn, cid, st.session_state.username, "decision",
                      f"{decision}" + (f" (model verdict overridden: {reason.strip()})" if override else ""))
            st.success(f"Claim #{cid} marked '{decision}'.")
            st.rerun()

    st.divider()
    st.subheader("Investigator notes")
    with st.form(f"note_form_{cid}", clear_on_submit=True):
        new_note = st.text_area("Add a note (internal - not visible to providers)")
        if st.form_submit_button("Save note") and new_note.strip():
            log_event(conn, cid, st.session_state.username, "note", new_note.strip())
            st.rerun()

    ev_rows = conn.execute(
        "SELECT created_at, actor, event_type, detail FROM claim_events WHERE claim_id = ? ORDER BY id",
        (cid,),
    ).fetchall()
    st.subheader("Review timeline")
    st.write(f"- `{submitted_at}`  **submitted** by {submitted_by} ({source}) - scored {label}")
    if not ev_rows:
        st.caption("No review activity recorded yet.")
    for t, actor, typ, detail in ev_rows:
        icon = "\U0001F4DD" if typ == "note" else "\u2696\ufe0f"
        st.write(f"- `{t}`  {icon} **{typ}** by {actor}: {detail}")

    report_text = build_claim_report(
        cid,
        {"Patient": patient_label, "Provider": f"{facility} (Provider {provider_id[:8]})",
         "Encounter date": service_date, "Diagnosis": condition, "Encounter type": encounter_type,
         "County": county, "Claim amount (KES)": f"{claim_amount:,.0f}",
         "Base cost (KES)": f"{base_cost:,.0f}", "Claim-to-base ratio": f"{ratio:.2f}",
         "Review status": review_status},
        {"XGBoost cost-anomaly score": f"{score:.3f}",
         "GraphSAGE relational score": f"{gs_score:.3f}" if gs_score is not None else "n/a",
         "Verdict": label, "Flagged by": flagged_by or "n/a"},
        report_flags, shap_lines,
        [(submitted_at, submitted_by, "submitted", f"scored {label}")] + [(t, a, ty, d) for t, a, ty, d in ev_rows],
        items=[(c, i, q, tp, bp) for c, i, q, tp, bp in item_rows],
    )
    st.download_button("Download claim report (.txt)", report_text,
                       file_name=f"claim_{cid}_report.txt", mime="text/plain")


# ============================================================
# FRAUD ANALYTICS OVERVIEW (auditor)
# ============================================================
def analytics_page(conn):
    st.header("Fraud analytics overview")
    df = pd.read_sql_query(
        "SELECT submitted_at, provider_id, fraud_score, risk_label, "
        "graphsage_score, flagged_by FROM claims_log",
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

    st.subheader("What's driving the flags - XGBoost vs GraphSAGE")
    st.caption(
        "Who actually raised the alarm on flagged claims (HIGH/MEDIUM only). This is "
        "the hybrid architecture's real contribution: claims GraphSAGE alone caught "
        "would have been missed by a cost-anomaly model working on its own."
    )

    def _flag_bucket(reason):
        if not isinstance(reason, str) or not reason:
            return "Neither"
        if "both models" in reason:
            return "Both models"
        if "GraphSAGE" in reason:
            return "GraphSAGE only"
        if "XGBoost" in reason:
            return "XGBoost only"
        return "Neither"

    flagged_for_breakdown = df[df["risk_label"].isin(["HIGH", "MEDIUM"])].copy()
    if flagged_for_breakdown.empty:
        st.caption("No flagged claims yet.")
    else:
        flagged_for_breakdown["flag_bucket"] = flagged_for_breakdown["flagged_by"].map(_flag_bucket)
        bucket_counts = (
            flagged_for_breakdown["flag_bucket"]
            .value_counts()
            .reindex(["XGBoost only", "GraphSAGE only", "Both models", "Neither"])
            .fillna(0)
        )
        bchart_col, bmetric_col = st.columns([2, 1])
        with bchart_col:
            st.bar_chart(bucket_counts)
        with bmetric_col:
            gs_only = int(bucket_counts.get("GraphSAGE only", 0))
            st.metric(
                "Caught by GraphSAGE alone",
                gs_only,
                help="Flagged claims that XGBoost's cost-anomaly score would NOT "
                     "have raised on its own - pure relational-pattern catches "
                     "(e.g. collusion rings) a tabular model misses.",
            )
            st.caption(
                f"{gs_only} of {len(flagged_for_breakdown)} flagged claims "
                f"({gs_only / len(flagged_for_breakdown) * 100:.1f}%) were caught only "
                "because of the relational (GraphSAGE) score."
            )

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

    st.subheader("Trends over time")
    trend = df.copy()
    trend["date"] = pd.to_datetime(trend["submitted_at"]).dt.date
    daily = trend.groupby("date").agg(
        claims=("fraud_score", "size"),
        avg_claim_amount_kes=("claim_amount_kes", "mean"),
        avg_fraud_score=("fraud_score", "mean"),
    )
    if len(daily) < 2:
        st.caption("Trend charts appear once this provider has claims on at least two different days.")
    else:
        t1, t2, t3 = st.columns(3)
        t1.caption("Claims per day"); t1.line_chart(daily["claims"])
        t2.caption("Avg claim amount (KES)"); t2.line_chart(daily["avg_claim_amount_kes"])
        t3.caption("Avg fraud score"); t3.line_chart(daily["avg_fraud_score"])
        base = artifacts["provider_mean_lookup"].get(provider_id, artifacts["global_mean"])
        st.caption(f"Training-data baseline for this provider: avg claim KES {base:,.0f}.")

    st.dataframe(df, use_container_width=True)
    st.download_button("Export this provider's claims (CSV)", df.to_csv(index=False),
                       file_name=f"provider_{provider_id[:8]}_claims.csv", mime="text/csv")


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
# MODEL HEALTH (admin) - surfaces the evaluation/stress-test reports already
# produced by the training scripts (per-fraud-type recall, the hybrid
# robustness stress test, tuned thresholds) directly in the app, rather than
# leaving them as one-off CSVs on disk that only exist in a terminal
# session. Read-only, no claims/patient data touched - fits admin's
# account-provisioning-only scope since it's model/system health, not
# clinical data.
# ============================================================
def model_health_page(gs_artifacts):
    st.header("Model health")
    st.caption(
        "Evaluation and stress-test results from the latest training/validation "
        "run - regenerated by re-running the training/evaluation scripts, not "
        "live-computed here."
    )

    health = load_model_health_artifacts()

    t_xgb = health["t_xgb"]
    t_gs = health["t_graphsage"]
    if t_xgb is not None and t_gs is not None:
        c1, c2 = st.columns(2)
        c1.metric("XGBoost auto-flag threshold", f"{t_xgb:.2f}")
        c2.metric("GraphSAGE auto-flag threshold", f"{t_gs:.2f}")
        st.caption(
            "Tuned via grid search on the validation split only (never test) to "
            "maximise F1 - see tune_hybrid_threshold.py."
        )
    else:
        st.caption("Tuned thresholds file not found - using defaults (0.5 / 0.5).")

    st.divider()
    st.subheader("Per-fraud-type recall - XGBoost alone (clean test set)")
    recall_df = health["per_fraudtype_recall"]
    if recall_df is None:
        st.caption("per_fraudtype_recall_v2data.csv not found - re-run xgboost_model_v2_robust.py to generate it.")
    else:
        display_recall = recall_df.rename(columns={
            "fraud_type": "Fraud type", "n": "Test-set count", "caught": "Caught",
            "recall": "Recall", "mean_score": "Mean score",
        })
        st.dataframe(display_recall, use_container_width=True, hide_index=True)
        st.caption(
            "This is XGBoost scored alone - the gap on collusion_ring is exactly "
            "why GraphSAGE exists (see the hybrid breakdown below and on the "
            "Analytics page)."
        )

    st.divider()
    st.subheader("Hybrid robustness / stress test")
    rob_df = health["hybrid_robustness"]
    if rob_df is None:
        st.caption("hybrid_robustness_results.csv not found - re-run hybrid_robustness_test.py to generate it.")
    else:
        st.caption(
            "Each row corrupts a different part of the pipeline (XGBoost's tabular "
            "features, GraphSAGE's node features, or simulated 'new to the system' "
            "claims with no graph history) and re-scores the identical held-out "
            "test set. Compares the tuned hybrid rule against XGBoost alone."
        )
        summary_cols = [
            "condition", "xgboost_alone_recall", "xgboost_alone_fp_rate",
            "hybrid_tuned_recall", "hybrid_tuned_precision", "hybrid_tuned_f1", "hybrid_tuned_fp_rate",
        ]
        available_cols = [c for c in summary_cols if c in rob_df.columns]
        display_rob = rob_df[available_cols].rename(columns={
            "condition": "Condition",
            "xgboost_alone_recall": "XGBoost recall", "xgboost_alone_fp_rate": "XGBoost FP rate",
            "hybrid_tuned_recall": "Hybrid recall", "hybrid_tuned_precision": "Hybrid precision",
            "hybrid_tuned_f1": "Hybrid F1", "hybrid_tuned_fp_rate": "Hybrid FP rate",
        })
        for pct_col in ["XGBoost recall", "XGBoost FP rate", "Hybrid recall", "Hybrid precision", "Hybrid F1", "Hybrid FP rate"]:
            if pct_col in display_rob.columns:
                display_rob[pct_col] = (display_rob[pct_col] * 100).round(2)
        st.dataframe(display_rob, use_container_width=True, hide_index=True)

        st.subheader("Hybrid recall by condition")
        chart_df = rob_df.set_index("condition")[["xgboost_alone_recall", "hybrid_tuned_recall"]]
        chart_df.columns = ["XGBoost alone", "Hybrid (tuned)"]
        st.bar_chart(chart_df)

        if "recall_collusion_ring_hybrid" in rob_df.columns:
            st.subheader("collusion_ring recall by condition (the fraud type GraphSAGE specialises in)")
            ring_chart = rob_df.set_index("condition")[["recall_collusion_ring_xgb", "recall_collusion_ring_hybrid"]]
            ring_chart.columns = ["XGBoost alone", "Hybrid (tuned)"]
            st.bar_chart(ring_chart)
            st.caption(
                "The clean-condition gap here (XGBoost alone ~0.8% vs hybrid 100%) is "
                "the headline result for this thesis's hybrid architecture. The "
                "'GraphSAGE node features corrupted: 10%' row shows where that "
                "advantage is fragile - a known, documented limitation, not something "
                "papered over."
            )

    st.divider()
    st.subheader("GraphSAGE coverage")
    if gs_artifacts is None:
        st.caption("GraphSAGE artifacts not loaded - app is running in XGBoost-only mode.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Patients with graph history", gs_artifacts.get("n_patients_with_history", "n/a"))
        c2.metric("Providers with graph history", gs_artifacts.get("n_providers_with_history", "n/a"))
        st.caption(
            "A patient or provider outside this coverage gets no relational score - "
            "the hybrid falls back to XGBoost alone for them, by design (see the "
            "'Missing GraphSAGE history' rows above for how much recall that costs)."
        )


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
    gs_artifacts = load_graphsage_artifacts()
    role = st.session_state.get("role", "nurse")

    st.sidebar.title("\U0001F6E1️ Fraud Detection")
    st.sidebar.caption(f"Signed in as **{st.session_state.username}**  ·  role: **{role}**")
    st.sidebar.toggle("\U0001F319 Dark mode", key="dark_mode")
    st.sidebar.divider()

    nav_options = []
    if role == "nurse":
        nav_options += ["Register patient", "Record encounter"]
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
        nav_options += ["Manage users", "Model Health"]

    page = st.sidebar.radio("Navigate", nav_options)
    if page != "Claims dashboard":
        st.session_state.selected_claim_id = None

    if st.sidebar.button("Log out"):
        st.session_state.logged_in = False
        st.rerun()

    if page == "Register patient":
        register_patient_page(conn, artifacts)
    elif page == "Record encounter":
        record_encounter_page(conn, artifacts)
    elif page == "Submit claim":
        submit_claim_page(conn, model, artifacts, gs_artifacts)
    elif page == "Upload CSV":
        upload_csv_page(conn, model, artifacts, gs_artifacts)
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
    elif page == "Model Health":
        model_health_page(gs_artifacts)
    else:
        manage_users_page(conn)


if __name__ == "__main__":
    main()
