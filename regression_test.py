import sys
sys.path.insert(0, "src")
import pandas as pd
import numpy as np

from type_cleaner import _parse_numeric_value, _match_year_canonical, _normalize_year_text
from ml_cleaner import _stem

failures = []

def check(label, condition):
    status = "PASS" if condition else "FAIL"
    if not condition:
        failures.append(label)
    print(f"[{status}] {label}")

# --- Salary /month, per year, monthly ---
check("salary /month strips suffix correctly", _parse_numeric_value("260,882/month") == 260882.0)
check("salary per year strips suffix correctly", _parse_numeric_value("50000 per year") == 50000.0)

# --- Currency conversion (OFF by default, opt-in via convert_currency=True) ---
check("Default: $ is NOT converted (just stripped)", _parse_numeric_value("$1,191.00") == 1191.0)
check("Default: EUR text is NOT converted", _parse_numeric_value("100 EUR") == 100.0)
check("Opt-in: USD $ converts to PKR", _parse_numeric_value("$1,191.00", convert_currency=True) == 333480.0)
check("Opt-in: EUR text code converts to PKR", _parse_numeric_value("100 EUR", convert_currency=True) == 30500.0)
check("Opt-in: GBP symbol converts to PKR", _parse_numeric_value("£100", convert_currency=True) == 35500.0)
check("Plain PKR stays unconverted either way", _parse_numeric_value("Rs. 5,000") == 5000.0)

# --- Numeric false positives ---
check("'Customer 1' is NOT parsed as a number", pd.isna(_parse_numeric_value("Customer 1")))

# --- GPA percentage conversion ---
from type_cleaner import standardize_gpa_columns
gpa_df = pd.DataFrame({"gpa": ["3.5", "90.2%", "-1.0", "5.5"]})
gpa_out = standardize_gpa_columns(gpa_df.copy())
check("GPA 90.2% converts to 4.0 scale", abs(gpa_out["gpa"].iloc[1] - 3.608) < 0.01)
check("GPA invalid values become NaN", pd.isna(gpa_out["gpa"].iloc[2]) and pd.isna(gpa_out["gpa"].iloc[3]))

# --- Gender standardization ---
from type_cleaner import standardize_gender_columns
gender_df = pd.DataFrame({"gender": ["F", "Female", "M", "Male"]})
gender_out = standardize_gender_columns(gender_df.copy())
check("Gender F/Female merge", gender_out["gender"].nunique() == 2)

# --- Year of study (exact + abbreviated) ---
for raw, expected in [("4th Year", "4th Year"), ("Fresh.", "1st Year"), ("Soph.", "2nd Year"),
                       ("Sr.", "4th Year"), ("Jr.", "3rd Year"), ("Freshman", "1st Year")]:
    result = _match_year_canonical(_normalize_year_text(raw))
    check(f"Year '{raw}' -> '{expected}'", result == expected)

# --- Major abbreviation expansion ---
check("Bba stems same as Business Administration", _stem("Bba") == _stem("Business Administration"))
check("Cs stems same as Computer Science", _stem("Cs") == _stem("Computer Science"))
check("Elec. Engg stems same as Electrical Engineering", _stem("Elec. Engg") == _stem("Electrical Engineering"))

# --- Banking abbreviations ---
check("Ib stems same as Internet Banking", _stem("Ib") == _stem("Internet Banking"))
check("Atm W/D stems same as Atm Withdrawal", _stem("Atm W/D") == _stem("Atm Withdrawal"))
check("MobileApp (no space) stems same-ignoring-space as Mobile App",
      _stem("MobileApp").replace(" ", "") == _stem("Mobile App").replace(" ", ""))
check("Bill Pmt stems same as Bill Payment", _stem("Bill Pmt") == _stem("Bill Payment"))

# --- Ops/HR/plural-singular business terms (stemmer fix) ---
check("Ops stems same as Operations", _stem("Ops") == _stem("Operations"))
check("Human Resource (singular) stems same as Human Resources (plural)",
      _stem("Human Resource") == _stem("Human Resources"))
check("HR stems same as Human Resources", _stem("HR") == _stem("Human Resources"))
check("Sales stems consistently", _stem("Sales") == _stem("sale"))
check("Cust. Support stems same as Customer Support", _stem("Cust. Support") == _stem("Customer Support"))

# --- Transitive grouping (union-find) ---
from ml_cleaner import fuzzy_standardize_categories
transitive_df = pd.DataFrame({"status": ["Dropped Out", "DroppedOut", "Drop-Out"] * 3})
transitive_out = fuzzy_standardize_categories(transitive_df.copy(), "status")
check("Transitive grouping merges Drop-Out/DroppedOut/Dropped Out all together",
      transitive_out["status"].nunique() == 1)

# --- Missing placeholders (context-aware) ---
from type_cleaner import standardize_missing_placeholders
status_df = pd.DataFrame({"employment_status": ["Active", "Pending"], "department": ["IT", "Pending"]})
status_out = standardize_missing_placeholders(status_df.copy())
check("'Pending' stays valid in a *_status column", status_out["employment_status"].iloc[1] == "Pending")
check("'Pending' becomes missing in a non-status column", pd.isna(status_out["department"].iloc[1]))

tbc_df = pd.DataFrame({"channel": ["ATM", "Tbc"]})
tbc_out = standardize_missing_placeholders(tbc_df.copy())
check("'Tbc' becomes missing", pd.isna(tbc_out["channel"].iloc[1]))

# --- Domain bounds: negative price/salary/fee, invalid age ---
from type_cleaner import enforce_domain_bounds
bounds_df = pd.DataFrame({
    "price": [100.0, -50.0],
    "salary": [50000.0, 0.0],
    "semester_fee": [30000.0, 0.0],
    "age": [25.0, -5.0],
})
bounds_out = enforce_domain_bounds(bounds_df.copy())
check("Negative price -> NaN", pd.isna(bounds_out["price"].iloc[1]))
check("Zero salary -> NaN", pd.isna(bounds_out["salary"].iloc[1]))
check("Zero semester_fee -> NaN", pd.isna(bounds_out["semester_fee"].iloc[1]))
check("Negative age -> NaN", pd.isna(bounds_out["age"].iloc[1]))

# --- ID-like / phone / currency columns excluded from wrong treatment ---
from type_cleaner import auto_detect_numeric_text_columns
id_df = pd.DataFrame({"customer_name": [f"Customer {i}" for i in range(20)], "phone": ["03001234567"]*20})
check("customer_name NOT detected as numeric", "customer_name" not in auto_detect_numeric_text_columns(id_df))
check("phone NOT detected as numeric (leading zero protection)", "phone" not in auto_detect_numeric_text_columns(id_df))

# --- Full pipeline end-to-end smoke test ---
from rule_cleaner import run_rule_based_cleaning
from ml_cleaner import run_ml_cleaning
smoke_df = pd.read_csv("data/sample_messy_data.csv")
smoke_clean = run_rule_based_cleaning(smoke_df.copy())
smoke_clean = run_ml_cleaning(smoke_clean)
check("Full pipeline runs without crashing", smoke_clean is not None and len(smoke_clean) > 0)

print()
print("=" * 50)
if failures:
    print(f"{len(failures)} REGRESSION(S) FOUND:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("ALL REGRESSION CHECKS PASSED")