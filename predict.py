from pathlib import Path

import joblib
import pandas as pd

# ==========================================
# Load Saved Files
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")


def normalize_choice(value):
    return " ".join(value.strip().lower().split())


def encode_choice(value, allowed_values, field_name):
    normalized = normalize_choice(value)
    if normalized not in allowed_values:
        print(f"\nInvalid {field_name}!")
        print("Allowed Values:", ", ".join(allowed_values.keys()))
        raise SystemExit(1)
    return allowed_values[normalized]


# ==========================================
# User Input
# ==========================================

print("\n========== Loan Approval Prediction ==========\n")

no_of_dependents = int(input("Number of Dependents: "))

education = input("Education (Graduate/Not Graduate): ")

self_employed = input("Self Employed (Yes/No): ")

income_annum = float(input("Annual Income: ").replace(",", ""))

loan_amount = float(input("Loan Amount: ").replace(",", ""))

loan_term = float(input("Loan Term (Years): ").replace(",", ""))

cibil_score = float(input("CIBIL Score: ").replace(",", ""))

residential_assets_value = float(
    input("Residential Assets Value: ").replace(",", "")
)

commercial_assets_value = float(
    input("Commercial Assets Value: ").replace(",", "")
)

luxury_assets_value = float(
    input("Luxury Assets Value: ").replace(",", "")
)

bank_asset_value = float(
    input("Bank Asset Value: ").replace(",", "")
)

# ==========================================
# Encode Categorical Values
# ==========================================

education = encode_choice(
    education,
    {
        "graduate": 0,
        "not graduate": 1,
    },
    "Education",
)

self_employed = encode_choice(
    self_employed,
    {
        "no": 0,
        "yes": 1,
    },
    "Self Employed value",
)

# ==========================================
# Create DataFrame
# ==========================================

data = pd.DataFrame({

    "no_of_dependents": [no_of_dependents],
    "education": [education],
    "self_employed": [self_employed],
    "income_annum": [income_annum],
    "loan_amount": [loan_amount],
    "loan_term": [loan_term],
    "cibil_score": [cibil_score],
    "residential_assets_value": [residential_assets_value],
    "commercial_assets_value": [commercial_assets_value],
    "luxury_assets_value": [luxury_assets_value],
    "bank_asset_value": [bank_asset_value]

})

# ==========================================
# Scale
# ==========================================

data_scaled = scaler.transform(data)

# ==========================================
# Prediction
# ==========================================

prediction = model.predict(data_scaled)[0]
prediction = "Approved" if int(prediction) == 0 else "Rejected"

print("\n======================================")

print("Prediction :", prediction)

if prediction.strip().lower() == "approved":
    print("Loan Approved")
else:
    print("Loan Rejected")

print("======================================")
