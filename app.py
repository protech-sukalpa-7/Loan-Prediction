from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
FEATURE_COLUMNS = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

CHOICE_ENCODERS = {
    "education": {
        "graduate": 0,
        "not graduate": 1,
    },
    "self_employed": {
        "no": 0,
        "yes": 1,
    },
}

model = joblib.load(BASE_DIR / "model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")

app = Flask(__name__)


def normalize_choice(value):
    return " ".join(value.strip().lower().split())


def encode_choice(field_name, value):
    choices = CHOICE_ENCODERS[field_name]
    normalized = normalize_choice(value)
    if normalized not in choices:
        allowed = ", ".join(choices.keys())
        raise ValueError(f"Invalid {field_name}. Allowed values: {allowed}")
    return choices[normalized]


def parse_float(form, field_name):
    value = form.get(field_name, "").replace(",", "").strip()
    if value == "":
        raise ValueError(f"{field_name} is required")
    return float(value)


def build_input_frame(form):
    row = {
        "no_of_dependents": int(parse_float(form, "no_of_dependents")),
        "education": encode_choice("education", form.get("education", "")),
        "self_employed": encode_choice("self_employed", form.get("self_employed", "")),
        "income_annum": parse_float(form, "income_annum"),
        "loan_amount": parse_float(form, "loan_amount"),
        "loan_term": parse_float(form, "loan_term"),
        "cibil_score": parse_float(form, "cibil_score"),
        "residential_assets_value": parse_float(form, "residential_assets_value"),
        "commercial_assets_value": parse_float(form, "commercial_assets_value"),
        "luxury_assets_value": parse_float(form, "luxury_assets_value"),
        "bank_asset_value": parse_float(form, "bank_asset_value"),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_loan_status(form):
    data = build_input_frame(form)
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    return "Approved" if int(prediction) == 0 else "Rejected"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        prediction = predict_loan_status(request.form)
    except ValueError as error:
        prediction = f"Error: {error}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
