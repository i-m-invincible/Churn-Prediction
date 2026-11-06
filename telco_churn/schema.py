"""The raw input contract shared by training, inference and the dashboard."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco.csv"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
SEED = 42
THRESHOLD = 0.5
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
INTERNET_ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
CATEGORIES = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["0", "1"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    **{name: ["No", "Yes", "No internet service"] for name in INTERNET_ADDONS},
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}
CATEGORICAL_FEATURES = list(CATEGORIES)
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
EXPLANATION_GROUPS = {
    "Tenure": ["tenure"],
    "Monthly charges": ["MonthlyCharges"],
    "Total charges": ["TotalCharges"],
    "Contract": ["Contract"],
    "Payment method": ["PaymentMethod"],
    "Paperless billing": ["PaperlessBilling"],
    "Phone services": ["PhoneService", "MultipleLines"],
    "Internet services": ["InternetService", *INTERNET_ADDONS],
    "Gender": ["gender"],
    "Senior citizen": ["SeniorCitizen"],
    "Partner": ["Partner"],
    "Dependents": ["Dependents"],
}
