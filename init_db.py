"""
Standalone SQLite database setup for the Streamlit fraud-scoring dashboard.

Creates claims_dashboard_v2.db with two tables:
  - users: login accounts (username, sha256 password hash)
  - claims_log: every claim scored through the app (manual or CSV), with its
    fraud score and risk label

Safe to re-run: uses CREATE TABLE IF NOT EXISTS and only seeds the demo
admin account if the users table is empty. app.py connects to this same
file (get_db()) and will find these tables already there - this script
just makes the "create the database" step explicit and independently
runnable/inspectable, rather than something that only happens implicitly
on first app launch.
"""

import os
import sqlite3
import hashlib

DB_PATH = os.path.expanduser("~/synthea_kenya/output/model/claims_dashboard_v2.db")


def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def main():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    # Same locking workaround as app.py - the connected folder doesn't
    # support SQLite's normal POSIX file locking.
    conn.execute("PRAGMA journal_mode=MEMORY")
    conn.execute("PRAGMA synchronous=OFF")

    # Schema matches app.py's get_db() exactly (no extra columns) so this
    # script and the app always agree on structure - no migration needed
    # for the users table already created on first app launch.
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

    cur = conn.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            ("admin", hash_password("admin123")),
        )
        conn.commit()
        print("Seeded demo admin account: admin / admin123")
    else:
        print("Users table already populated - not re-seeding.")

    users = conn.execute("SELECT username FROM users").fetchall()
    print(f"\nDatabase ready at {DB_PATH}")
    print(f"Users ({len(users)}):")
    for (username,) in users:
        print(f"  - {username}{' (admin)' if username == 'admin' else ''}")

    claims_count = conn.execute("SELECT COUNT(*) FROM claims_log").fetchone()[0]
    print(f"\nclaims_log: {claims_count} rows logged so far")
    conn.close()


if __name__ == "__main__":
    main()
