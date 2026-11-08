"""Automated tests for the fraud detection dashboard.

Run from the project folder:   python -m pytest tests -v

Unit tests exercise single functions; the integration tests at the bottom drive
the real Streamlit app headlessly (streamlit.testing AppTest) against a
throw-away SQLite database, so your real claims database is never touched.
"""
import os, sys, shutil, sqlite3, datetime as dt, tempfile
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("FRAUD_DB_PATH", os.path.join(tempfile.mkdtemp(), "unit.db"))

import app                      # noqa: E402  (module-level code only injects CSS)
import claim_reference as cref  # noqa: E402

ARTIFACTS = {"t_high": 0.7073, "t_low": 0.0468}
GS = {"t_graphsage": 0.95}


# ---------------------------------------------------------------- passwords
def test_password_hash_roundtrip():
    h, salt = app.hash_password("Demo@2026")
    assert app.verify_password("Demo@2026", h, salt)

def test_password_wrong_is_rejected():
    h, salt = app.hash_password("Demo@2026")
    assert not app.verify_password("demo@2026", h, salt)

def test_same_password_gets_different_salt_and_hash():
    h1, s1 = app.hash_password("same-password")
    h2, s2 = app.hash_password("same-password")
    assert s1 != s2 and h1 != h2

def test_legacy_unsalted_hash_still_verifies():
    import hashlib
    legacy = hashlib.sha256(b"old-pass").hexdigest()
    assert app.verify_password("old-pass", legacy, None)


# ---------------------------------------------------------------- age / labels
def test_calculate_age_before_and_after_birthday():
    assert app.calculate_age(dt.date(2000, 6, 15), dt.date(2026, 6, 14)) == 25
    assert app.calculate_age(dt.date(2000, 6, 15), dt.date(2026, 6, 15)) == 26

@pytest.mark.parametrize("p,expected", [(0.9, "HIGH"), (0.7073, "HIGH"), (0.3, "MEDIUM"), (0.0468, "MEDIUM"), (0.01, "LOW")])
def test_risk_label_thresholds(p, expected):
    assert app.risk_label(p, ARTIFACTS) == expected


# ---------------------------------------------------------------- hybrid rule
def test_hybrid_graphsage_only_is_high_and_says_so():
    label, why = app.hybrid_label(0.03, 0.998, ARTIFACTS, GS)
    assert label == "HIGH" and "GraphSAGE" in why

def test_hybrid_both_models_flag():
    label, why = app.hybrid_label(0.95, 0.99, ARTIFACTS, GS)
    assert label == "HIGH" and why.startswith("both models")

def test_hybrid_xgboost_only_is_high():
    label, why = app.hybrid_label(0.95, 0.10, ARTIFACTS, GS)
    assert label == "HIGH" and "XGBoost" in why

def test_hybrid_mid_score_is_medium():
    assert app.hybrid_label(0.30, 0.10, ARTIFACTS, GS)[0] == "MEDIUM"

def test_hybrid_low_when_nothing_fires():
    assert app.hybrid_label(0.01, 0.10, ARTIFACTS, GS)[0] == "LOW"

def test_hybrid_unknown_patient_uses_xgboost_alone():
    # GraphSAGE returns None for patients/providers not in the saved graph
    assert app.hybrid_label(0.95, None, ARTIFACTS, GS)[0] == "HIGH"
    assert app.hybrid_label(0.01, None, ARTIFACTS, GS)[0] == "LOW"

def test_hybrid_provider_cluster_fallback_raises_to_medium(monkeypatch):
    monkeypatch.setattr(app, "provider_cluster_exposure", lambda pid: 0.8)
    label, why = app.hybrid_label(0.01, None, ARTIFACTS, GS, provider_id="ring-provider")
    assert label == "MEDIUM" and "cluster" in why

def test_hybrid_provider_cluster_below_minimum_is_ignored(monkeypatch):
    monkeypatch.setattr(app, "provider_cluster_exposure", lambda pid: 0.2)
    assert app.hybrid_label(0.01, None, ARTIFACTS, GS, provider_id="p")[0] == "LOW"


# ---------------------------------------------------------------- department log
S, E = dt.datetime(2026, 11, 2, 9, 0), dt.datetime(2026, 11, 2, 17, 0)

def _df(rows):
    return pd.DataFrame(rows, columns=["Department", "Start", "End"])

def test_default_flow_stays_inside_visit_and_never_overlaps():
    rows = app.default_department_rows("inpatient", S, E)
    assert rows[0][1] == S and rows[-1][2] <= E
    assert all(a[2] <= b[1] for a, b in zip(rows, rows[1:]))

def test_department_check_accepts_valid_log():
    rows, errs = app.check_departments(_df([("Triage", S, S + dt.timedelta(minutes=15)),
                                            ("Laboratory", S + dt.timedelta(minutes=15), S + dt.timedelta(hours=1))]), S, E)
    assert errs == [] and len(rows) == 2

def test_department_check_rejects_overlap():
    _, errs = app.check_departments(_df([("Triage", S, S + dt.timedelta(minutes=30)),
                                         ("Pharmacy", S + dt.timedelta(minutes=20), S + dt.timedelta(minutes=50))]), S, E)
    assert any("overlap" in e for e in errs)

def test_department_check_rejects_time_outside_visit():
    _, errs = app.check_departments(_df([("Ward", S - dt.timedelta(hours=1), S + dt.timedelta(hours=1))]), S, E)
    assert any("inside the visit" in e for e in errs)

def test_department_check_rejects_end_before_start():
    _, errs = app.check_departments(_df([("Triage", S + dt.timedelta(hours=1), S)]), S, E)
    assert any("end must be after start" in e for e in errs)

def test_department_check_requires_at_least_one_row():
    assert app.check_departments(_df([]), S, E)[1]

def test_department_hours_add_up_to_total_stay():
    rows = [("Triage", S, S + dt.timedelta(minutes=15)), ("Ward", S + dt.timedelta(hours=1), E)]
    tbl = app.department_summary_df(rows, 8.0)
    lines, total = tbl[tbl.Department != "Total"], tbl[tbl.Department == "Total"].Hours.iloc[0]
    assert abs(lines.Hours.sum() - total) < 0.011 and abs(total - 8.0) < 0.011
    assert "Waiting / between departments" in set(tbl.Department)


# ---------------------------------------------------------------- tariff reference
def test_every_catalog_item_has_positive_price():
    assert all(cref.tariff_price(n) > 0 for n in cref.ITEM_NAMES)

def test_payer_coverage_never_exceeds_claim():
    for scheme in cref.SCHEME_CHOICES:
        assert cref.payer_coverage("inpatient", scheme, 50000, 30000) <= 30000

def test_insurer_factors_are_sensible():
    assert all(0.5 <= f <= 1.0 for f in cref.INSURER_FACTORS.values()) and len(cref.INSURER_FACTORS) == 12


# ---------------------------------------------------------------- model + database
@pytest.fixture(scope="module")
def model_and_artifacts():
    return app.load_model(), app.load_artifacts()

def _scored(model, artifacts, ratio, conn):
    raw = {"PROVIDER_ID": next(iter(artifacts["provider_org_lookup"])), "PRIMARY_CONDITION": artifacts["conditions"][0],
           "PATIENT_COUNTY": "Nairobi", "ENCOUNTER_TYPE": "outpatient", "GENDER": "F", "SERVICE_DATE": "2026-10-05",
           "AGE": 40, "LENGTH_OF_STAY_HOURS": 4.0, "BASE_COST_KES": 3000.0, "CLAIM_AMOUNT_KES": 3000.0 * ratio,
           "PAYER_COVERAGE_KES": 2500.0, "CLAIM_TO_BASE_RATIO": ratio}
    feat = app.engineer_row(raw, artifacts, conn)
    return app.score_row(feat, model, artifacts)[0]

def test_model_scores_heavy_upcoding_far_above_a_normal_claim(model_and_artifacts):
    model, artifacts = model_and_artifacts
    conn = app.get_db()
    assert _scored(model, artifacts, 3.5, conn) > 0.7073 > _scored(model, artifacts, 1.4, conn)

# ---------------------------------------------------------------- unread badges
@pytest.mark.parametrize("n,expected", [(0, ""), (1, "1"), (12, "12"), (99, "99"), (100, "99+"), (5000, "99+")])
def test_badge_text_formats_the_count(n, expected):
    out = app.badge_text(n)
    assert (out == "") if expected == "" else (expected in out)

def test_nav_label_badges_only_the_page_that_owns_the_stream():
    counts = {"p": 2, "c": 7}
    assert "7" in app.nav_label("Claims dashboard", counts, "provider")
    assert "2" in app.nav_label("Submit claim", counts, "provider")
    assert app.nav_label("Provider profile", counts, "provider") == "Provider profile"
    assert app.nav_label("Upload CSV", counts, "provider") == "Upload CSV"

def test_nav_label_is_untouched_when_nothing_is_new():
    for page in ("Claims dashboard", "Submit claim", "Register patient"):
        assert app.nav_label(page, {"p": 0, "c": 0}, "provider") == page

def test_auditor_has_no_patient_badge_only_claims():
    assert app.nav_label("Claims dashboard", {"c": 4}, "auditor").startswith("Claims dashboard")
    assert "4" in app.nav_label("Claims dashboard", {"c": 4}, "auditor")
    assert "p" not in app.BADGE_OWNER_PAGE["auditor"]

def test_admin_role_carries_no_badges_at_all():
    assert app.BADGE_OWNER_PAGE.get("admin", {}) == {}

def test_database_creates_every_table():
    conn = app.get_db()
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"users", "patients", "encounters", "encounter_items", "encounter_departments",
            "claim_departments", "claim_items", "claims_log", "claim_events"} <= names


# ================================================================ integration (AppTest)
from streamlit.testing.v1 import AppTest   # noqa: E402

@pytest.fixture()
def fresh_app(tmp_path, monkeypatch):
    """The real app, pointed at an empty throw-away database (demo accounts are seeded on first run)."""
    monkeypatch.setenv("FRAUD_DB_PATH", str(tmp_path / "test.db"))
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=60)
    at.run()
    return at, tmp_path / "test.db"

def _login(at, user, pw="Demo@2026"):
    at.text_input[0].set_value(user); at.text_input[1].set_value(pw)
    at.button[0].click(); at.run()
    return at

def _nav(at):
    return list(at.sidebar.radio[0].options)

def test_wrong_password_is_refused(fresh_app):
    at, _ = fresh_app
    _login(at, "nurse_nyalenda", "not-the-password")
    assert at.error and not at.session_state["logged_in"] if "logged_in" in at.session_state else True
    assert "logged_in" not in at.session_state or not at.session_state["logged_in"]

def test_nurse_only_sees_nurse_pages(fresh_app):
    at, _ = fresh_app
    assert _nav(_login(at, "nurse_nyalenda")) == ["Register patient", "Record encounter", "Change password"]

def test_provider_cannot_see_review_or_admin_pages(fresh_app):
    at, _ = fresh_app
    nav = _nav(_login(at, "nyalenda_hospital"))
    assert "Submit claim" in nav and "Manage users" not in nav and "Audit history" not in nav

def test_provider_dashboard_only_lists_own_facility_claims(fresh_app):
    at, db = fresh_app
    conn = sqlite3.connect(db)
    mine = conn.execute("SELECT provider_id FROM users WHERE username='nyalenda_hospital'").fetchone()[0]
    other = conn.execute("SELECT provider_id FROM users WHERE username='kajiado_health_centre'").fetchone()[0]
    for pid in (mine, mine, other):
        conn.execute("INSERT INTO claims_log(submitted_at, submitted_by, source, provider_id, claim_amount_kes, "
                     "fraud_score, risk_label, review_status) VALUES ('2026-11-02T10:00:00','t','csv',?,1000,0.9,'HIGH','pending')", (pid,))
    conn.commit(); conn.close()
    _login(at, "nyalenda_hospital")
    at.sidebar.radio[0].set_value("Claims dashboard"); at.run()
    assert at.metric[0].value == "2"      # total claims visible to this facility

def test_auditor_sees_all_facilities(fresh_app):
    at, db = fresh_app
    conn = sqlite3.connect(db)
    pids = [r[0] for r in conn.execute("SELECT DISTINCT provider_id FROM users WHERE provider_id IS NOT NULL LIMIT 3")]
    for pid in pids:
        conn.execute("INSERT INTO claims_log(submitted_at, submitted_by, source, provider_id, claim_amount_kes, "
                     "fraud_score, risk_label, review_status) VALUES ('2026-11-02T10:00:00','t','csv',?,1000,0.9,'HIGH','pending')", (pid,))
    # an auditor account for the test
    h, salt = app.hash_password("Demo@2026")
    conn.execute("INSERT INTO users(username,password_hash,salt,role) VALUES ('aud_test',?,?, 'auditor')", (h, salt))
    conn.commit(); conn.close()
    _login(at, "aud_test")
    at.sidebar.radio[0].set_value("Claims dashboard"); at.run()
    assert at.metric[0].value == str(len(pids))


def _add_auditor(db, username="aud_badge"):
    conn = sqlite3.connect(db)
    h, salt = app.hash_password("Demo@2026")
    conn.execute("INSERT INTO users(username,password_hash,salt,role) VALUES (?,?,?,'auditor')", (username, h, salt))
    conn.commit(); conn.close()

def _add_claims(db, n, by="someone_else", pid="P1"):
    conn = sqlite3.connect(db)
    for _ in range(n):
        conn.execute("INSERT INTO claims_log(submitted_at, submitted_by, source, provider_id, claim_amount_kes, "
                     "fraud_score, risk_label, review_status) VALUES "
                     "('2026-11-06T10:00:00',?,'csv',?,1000,0.9,'HIGH','pending')", (by, pid))
    conn.commit(); conn.close()

def test_new_claims_raise_a_badge_and_opening_the_page_clears_it(fresh_app):
    at, db = fresh_app
    _add_auditor(db)
    _login(at, "aud_badge")
    at.sidebar.radio[0].set_value("Audit history"); at.run()   # sit away from the claims page
    _add_claims(db, 3)
    at.run()
    assert any("3" in o and o.startswith("Claims dashboard") for o in _nav(at)), _nav(at)
    at.sidebar.radio[0].set_value("Claims dashboard"); at.run()
    assert _nav(at)[0] == "Claims dashboard"                   # opening it clears the count

def test_claims_already_there_at_login_are_not_counted_as_new(fresh_app):
    at, db = fresh_app
    _add_auditor(db); _add_claims(db, 4)
    _login(at, "aud_badge")
    at.sidebar.radio[0].set_value("Audit history"); at.run()
    assert _nav(at) == ["Claims dashboard", "Audit history", "Analytics", "Provider profile", "Change password"]

def test_my_own_claims_never_badge_me(fresh_app):
    at, db = fresh_app
    _add_auditor(db)
    _login(at, "aud_badge")
    at.sidebar.radio[0].set_value("Audit history"); at.run()
    _add_claims(db, 5, by="aud_badge")
    at.run()
    assert _nav(at)[0] == "Claims dashboard"

def test_a_new_patient_badges_the_nurse_register_page(fresh_app):
    at, db = fresh_app
    _login(at, "nurse_nyalenda")
    at.sidebar.radio[0].set_value("Record encounter"); at.run()
    conn = sqlite3.connect(db)
    fac = conn.execute("SELECT provider_id FROM users WHERE username='nurse_nyalenda'").fetchone()[0]
    h, salt = app.hash_password("Demo@2026")
    conn.execute("INSERT INTO users(username,password_hash,salt,role,provider_id) VALUES "
                 "('nurse_two',?,?,'nurse',?)", (h, salt, fac))
    conn.execute("INSERT INTO patients(patient_id, first_name, last_name, gender, dob, age, county, "
                 "registered_by, registered_at, scheme) VALUES "
                 "('PX1','Amina','Otieno','F','1990-01-01',36,'Kisumu','nurse_two','2026-11-06T10:00:00','SHA')")
    conn.commit(); conn.close()
    at.run()
    assert any("1" in o and o.startswith("Register patient") for o in _nav(at)), _nav(at)
    at.sidebar.radio[0].set_value("Register patient"); at.run()
    assert _nav(at)[0] == "Register patient"

def test_a_patient_from_another_facility_does_not_badge_this_nurse(fresh_app):
    at, db = fresh_app
    _login(at, "nurse_nyalenda")
    at.sidebar.radio[0].set_value("Record encounter"); at.run()
    conn = sqlite3.connect(db)
    other = conn.execute("SELECT provider_id FROM users WHERE username='kajiado_health_centre'").fetchone()[0]
    h, salt = app.hash_password("Demo@2026")
    conn.execute("INSERT INTO users(username,password_hash,salt,role,provider_id) VALUES "
                 "('nurse_far',?,?,'nurse',?)", (h, salt, other))
    conn.execute("INSERT INTO patients(patient_id, first_name, last_name, gender, dob, age, county, "
                 "registered_by, registered_at, scheme) VALUES "
                 "('PX2','Said','Juma','M','1988-01-01',38,'Kajiado','nurse_far','2026-11-06T10:00:00','SHA')")
    conn.commit(); conn.close()
    at.run()
    assert _nav(at) == ["Register patient", "Record encounter", "Change password"]
