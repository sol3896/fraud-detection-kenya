"""
ILLUSTRATIVE reference data for the itemised claim form (app.py).

None of these prices, bed counts or scheme rules are real SHA / KEMSA /
insurer data - they are plausible demo values so the claim form can show a
realistic itemised bill. Replace the tables below with real tariffs,
formulary prices and facility records for any real deployment.

Calibration: the bundles and benefit rates are set so a typical itemised
claim lands in the same cost range the XGBoost model was trained on
(median legitimate base cost per encounter type), and the payer-coverage
rate per encounter type is the median coverage/base-cost ratio observed in
the training data (capped at 100% of tariff). That keeps the three cost
features the model reads (base cost, claim amount, payer coverage) in
distribution; it is not a claim that real schemes pay at these rates.
"""
import hashlib
import math

# (category, item, unit, tariff price KES, basis)
# basis: SHA = published SHA/SHIF rate; PUBLIC = public-facility service charter;
#        PROC = KNH procurement unit cost (FY2020/21) x a handling/dispensing markup;
#        MARKET = published market range (mostly private/mid-tier);  EST = estimate (no source found)
CATALOG = [
    ("Consultation", "Teleconsultation", "visit", 700, "EST"),
    ("Consultation", "Wellness screening package", "package", 1500, "EST"),
    ("Consultation", "General practitioner consultation", "visit", 800, "PUBLIC"),
    ("Consultation", "Specialist consultation", "visit", 1500, "PUBLIC"),
    ("Consultation", "Urgent care consultation", "visit", 1500, "EST"),
    ("Consultation", "Emergency department consultation", "visit", 1500, "PUBLIC"),
    ("Consultation", "Inpatient admission consultation", "visit", 1000, "PUBLIC"),
    ("Consultation", "Specialist ward review", "visit", 1000, "PUBLIC"),
    ("Consultation", "Home visit (clinician)", "visit", 3000, "EST"),
    ("Consultation", "Palliative care consultation", "visit", 1500, "EST"),
    ("Consultation", "Nursing care review", "visit", 500, "EST"),
    ("Medication", "Paracetamol 500mg", "tablet", 3, "PROC"),
    ("Medication", "Amoxicillin 500mg", "capsule", 12, "PROC"),
    ("Medication", "Artemether-Lumefantrine (full course)", "course", 350, "EST"),
    ("Medication", "Ceftriaxone 1g injection", "vial", 500, "PROC"),
    ("Medication", "Normal saline 1L", "bag", 150, "EST"),
    ("Medication", "Metformin 500mg", "tablet", 3, "PROC"),
    ("Medication", "Amlodipine 5mg", "tablet", 4, "PROC"),
    ("Medication", "Salbutamol inhaler", "inhaler", 450, "PROC"),
    ("Medication", "Omeprazole 20mg", "capsule", 10, "PROC"),
    ("Medication", "Morphine 10mg injection", "ampoule", 250, "PROC"),
    ("Medication", "Enoxaparin 40mg injection", "syringe", 1000, "PROC"),
    ("Medication", "Insulin glargine pen", "pen", 1800, "PROC"),
    ("Medication", "Oxytocin 10IU injection", "ampoule", 150, "PROC"),
    ("Medication", "Tetanus toxoid vaccine", "dose", 200, "PROC"),
    ("Laboratory", "Full blood count", "test", 600, "PUBLIC"),
    ("Laboratory", "Malaria rapid test", "test", 200, "PUBLIC"),
    ("Laboratory", "Urea & electrolytes", "test", 1500, "PUBLIC"),
    ("Laboratory", "Blood culture", "test", 2500, "EST"),
    ("Laboratory", "Laboratory panel (extended)", "panel", 2500, "EST"),
    ("Imaging", "Chest X-ray", "scan", 2000, "MARKET"),
    ("Imaging", "Abdominal ultrasound", "scan", 2500, "EST"),
    ("Imaging", "ECG", "test", 1500, "EST"),
    ("Imaging", "CT scan", "scan", 9000, "SHA/MARKET"),
    ("Imaging", "MRI scan", "scan", 20000, "SHA/MARKET"),
    ("Procedure", "Wound suturing / minor procedure", "procedure", 1500, "EST"),
    ("Procedure", "Nursing procedure", "procedure", 500, "EST"),
    ("Procedure", "Physiotherapy session", "session", 900, "PUBLIC"),
    ("Procedure", "Haemodialysis session", "session", 10650, "SHA"),
    ("Procedure", "Normal delivery (incl. newborn care)", "delivery", 10000, "SHA"),
    ("Procedure", "Caesarean section (incl. newborn care)", "surgery", 30000, "SHA"),
    ("Procedure", "Appendectomy", "surgery", 35000, "MARKET"),
    ("Procedure", "Hernia repair", "surgery", 50000, "MARKET"),
    ("Procedure", "Cholecystectomy", "surgery", 75000, "MARKET"),
    ("Procedure", "Theatre fee (major surgery)", "procedure", 25000, "EST"),
    ("Procedure", "Ambulance (within 25 km)", "trip", 4500, "SHA"),
]
BASIS_LABELS = {
    "SHA": "Published SHA/SHIF rate", "PUBLIC": "Public-facility service charter",
    "PROC": "KNH procurement cost + handling markup", "MARKET": "Published market range",
    "SHA/MARKET": "SHA rate / market range", "EST": "Estimate (no source found)",
}
CATALOG_BY_ITEM = {c[1]: c for c in CATALOG}


def price_basis(item):
    c = CATALOG_BY_ITEM.get(item)
    return BASIS_LABELS.get(c[4], c[4]) if c else ""
ITEM_NAMES = [c[1] for c in CATALOG]


def tariff_price(item):
    c = CATALOG_BY_ITEM.get(item)
    return float(c[3]) if c else 0.0


def item_category(item):
    c = CATALOG_BY_ITEM.get(item)
    return c[0] if c else "Other"


# Ward daily rates (tariff, KES per bed-day) - inpatient accommodation.
WARD_RATES = {"General ward": 3360, "Private ward": 15000, "High dependency (HDU)": 28000, "ICU": 45000}

# Default itemised bundle suggested per encounter type: [(item, qty), ...]
DEFAULT_BUNDLES = {
    "virtual": [("Teleconsultation", 1), ("Paracetamol 500mg", 20), ("Amoxicillin 500mg", 21)],
    "wellness": [("Wellness screening package", 1)],
    "ambulatory": [("Specialist consultation", 1), ("Full blood count", 1), ("Malaria rapid test", 1),
                   ("Paracetamol 500mg", 20), ("Omeprazole 20mg", 14)],
    "outpatient": [("General practitioner consultation", 1), ("Full blood count", 1), ("Chest X-ray", 1),
                   ("ECG", 1), ("Amoxicillin 500mg", 21)],
    "urgentcare": [("Urgent care consultation", 1), ("Wound suturing / minor procedure", 1),
                   ("Tetanus toxoid vaccine", 1), ("Chest X-ray", 1)],
    "emergency": [("Emergency department consultation", 1), ("ECG", 1), ("Laboratory panel (extended)", 1),
                  ("CT scan", 1), ("Ceftriaxone 1g injection", 4), ("Normal saline 1L", 4)],
    "inpatient": [("Inpatient admission consultation", 1), ("Specialist ward review", 2), ("Full blood count", 1),
                  ("Urea & electrolytes", 1), ("Laboratory panel (extended)", 1), ("CT scan", 1),
                  ("Ceftriaxone 1g injection", 8), ("Normal saline 1L", 8), ("Enoxaparin 40mg injection", 5)],
    "snf": [("Nursing care review", 10), ("Physiotherapy session", 1)],
    "home": [("Home visit (clinician)", 1), ("Nursing procedure", 2), ("Ceftriaxone 1g injection", 3)],
    "hospice": [("Palliative care consultation", 2), ("Morphine 10mg injection", 10)],
}

# Share of the tariff (base) cost the scheme pays, per encounter type.
# Derived from the training data's median coverage/base ratio (capped at 1.0).
BENEFIT_RATE = {"ambulatory": 1.0, "emergency": 1.0, "home": 0.83, "hospice": 0.84, "inpatient": 0.24,
                "outpatient": 0.31, "snf": 0.85, "urgentcare": 0.25, "virtual": 1.0, "wellness": 1.0}

# Scheme multipliers on that rate. Illustrative only.
SCHEMES = {"SHA (Social Health Authority)": 1.0, "Private insurer - comprehensive": 1.0,
           "Private insurer - basic": 0.8}
DEFAULT_SCHEME = "SHA (Social Health Authority)"


def payer_coverage(encounter_type, scheme, base_cost, claim_amount):
    """Coverage is driven by the TARIFF (base) cost - the allowed amount - not
    by whatever the provider chose to bill, and never exceeds the claim."""
    rate = BENEFIT_RATE.get(encounter_type, 0.5) * SCHEMES.get(scheme, 1.0)
    return round(min(claim_amount, base_cost * rate), 2)


def bed_days(length_of_stay_hours):
    return max(1, math.ceil(length_of_stay_hours / 24.0)) if length_of_stay_hours > 0 else 1


def facility_beds(provider_id):
    """Illustrative, deterministic bed capacity and typical occupancy for a
    facility, derived from a hash of its id (the dataset has no bed counts)."""
    h = int(hashlib.md5(str(provider_id).encode()).hexdigest()[:8], 16)
    capacity = 30 + h % 370
    base_occupancy_pct = 55 + (h // 7) % 26  # 55-80%
    return capacity, base_occupancy_pct


# Typical length of stay (hours) per encounter type = median in the training
# data. Used only to suggest a default visit end time, which staff can edit.
TYPICAL_LOS_HOURS = {"ambulatory": 14, "emergency": 14, "home": 12, "hospice": 570, "inpatient": 97,
                     "outpatient": 12, "snf": 367, "urgentcare": 12, "virtual": 12, "wellness": 13}
