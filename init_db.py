"""
Standalone SQLite database setup for the Streamlit fraud-scoring dashboard.

Creates claims_dashboard_v2.db with three tables:
  - users: login accounts (username, sha256 password hash, role - one of
    nurse/auditor/admin, controls which pages the account can see)
  - patients: patients registered through the Register patient page
    (patient_id, name, demographics) - optional link target for claims
  - claims_log: every claim scored through the app (manual or CSV), with its
    fraud score, risk label, optional linked patient_id, and review status
    (pending/approved/rejected/escalated) set from the Claim Detail page

Safe to re-run: uses CREATE TABLE IF NOT EXISTS, migrates an older DB file
that predates the role/patient/review columns by adding them with ALTER
TABLE, and only seeds the demo admin account if the users table is empty.
app.py connects to this same file (get_db(), which runs the identical
migration) and will find everything already there - this script just makes
the "create the database" step explicit and independently runnable/
inspectable, rather than something that only happens implicitly on first
app launch.
"""

import os
import sqlite3
import hashlib
import secrets

DB_PATH = os.path.expanduser("~/synthea_kenya/output/model/claims_dashboard_v2.db")


def hash_password(pw: str, salt: str | None = None) -> tuple[str, str]:
    """PBKDF2-HMAC-SHA256 with a random per-user salt - matches app.py's
    hash_password() exactly, so a DB this script creates/migrates and the
    app agree on format."""
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200_000).hex()
    return digest, salt


def main():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    # Same locking workaround as app.py - the connected folder doesn't
    # support SQLite's normal POSIX file locking.
    conn.execute("PRAGMA journal_mode=MEMORY")
    conn.execute("PRAGMA synchronous=OFF")

    # Schema matches app.py's get_db() exactly so this script and the app
    # always agree on structure.
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

    # Migration for a DB file created before role/patient/review columns existed.
    users_cols = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "role" not in users_cols:
        conn.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'nurse'")
        conn.execute("UPDATE users SET role = 'admin' WHERE username = 'admin'")
        conn.commit()
        print("Migrated users table: added 'role' column (admin account set to role=admin).")
    if "salt" not in users_cols:
        # Existing rows get salt = NULL (old unsalted SHA-256 hash) - app.py's
        # login flow verifies those the old way once, then transparently
        # re-hashes with a fresh salt on that successful login.
        conn.execute("ALTER TABLE users ADD COLUMN salt TEXT")
        conn.commit()
        print("Migrated users table: added 'salt' column (existing accounts upgrade on next login).")

    # One-time: "auditor" used to mean claim-submitter - rename those
    # existing accounts to "provider" (matches app.py's get_db() migration
    # exactly, same PRAGMA user_version guard so it only runs once).
    if conn.execute("PRAGMA user_version").fetchone()[0] < 1:
        changed = conn.execute("UPDATE users SET role = 'provider' WHERE role = 'auditor'").rowcount
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        if changed:
            print(f"Migrated {changed} account(s) from role 'auditor' to 'provider' "
                  "(auditor now means review-only).")

    patients_cols = {row[1] for row in conn.execute("PRAGMA table_info(patients)").fetchall()}
    if "dob" not in patients_cols:
        conn.execute("ALTER TABLE patients ADD COLUMN dob TEXT")
        conn.commit()
        print("Migrated patients table: added 'dob' column (new registrations compute age from it).")

    claims_cols = {row[1] for row in conn.execute("PRAGMA table_info(claims_log)").fetchall()}
    added = []
    for col, ddl in [
        ("review_status", "ALTER TABLE claims_log ADD COLUMN review_status TEXT DEFAULT 'pending'"),
        ("reviewed_by", "ALTER TABLE claims_log ADD COLUMN reviewed_by TEXT"),
        ("reviewed_at", "ALTER TABLE claims_log ADD COLUMN reviewed_at TEXT"),
        ("patient_id", "ALTER TABLE claims_log ADD COLUMN patient_id TEXT"),
        ("override_reason", "ALTER TABLE claims_log ADD COLUMN override_reason TEXT"),
        ("admission_datetime", "ALTER TABLE claims_log ADD COLUMN admission_datetime TEXT"),
        ("discharge_datetime", "ALTER TABLE claims_log ADD COLUMN discharge_datetime TEXT"),
    ]:
        if col not in claims_cols:
            conn.execute(ddl)
            added.append(col)
    if added:
        conn.commit()
        print(f"Migrated claims_log table: added columns {added}.")

    cur = conn.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        admin_hash, admin_salt = hash_password("admin123")
        conn.execute(
            "INSERT INTO users (username, password_hash, salt, role) VALUES (?, ?, ?, ?)",
            ("admin", admin_hash, admin_salt, "admin"),
        )
        conn.commit()
        print("Seeded demo admin account: admin / admin123 (role: admin)")
    else:
        print("Users table already populated - not re-seeding.")

    users = conn.execute("SELECT username, role FROM users").fetchall()
    print(f"\nDatabase ready at {DB_PATH}")
    print(f"Users ({len(users)}):")
    for username, role in users:
        print(f"  - {username}  (role: {role})")

    patients_count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    claims_count = conn.execute("SELECT COUNT(*) FROM claims_log").fetchone()[0]
    print(f"\npatients: {patients_count} registered so far")
    print(f"claims_log: {claims_count} rows logged so far")
    conn.close()


if __name__ == "__main__":
    main()
