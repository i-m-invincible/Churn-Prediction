"""Deterministic validation only: no fitted statistics or encodings belong here."""

from pathlib import Path

import numpy as np
import pandas as pd

from telco_churn.schema import CATEGORIES, DATA_PATH, FEATURES, INTERNET_ADDONS, NUMERIC_FEATURES


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate raw fields; leave blank TotalCharges for the training-fitted imputer."""
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("At least one customer is required.")
    result = frame.loc[:, FEATURES].copy()
    for name in NUMERIC_FEATURES:
        raw = result[name]
        blanks = raw.map(lambda value: isinstance(value, str) and not value.strip())
        raw = raw.mask(blanks, np.nan)
        converted = pd.to_numeric(raw, errors="coerce")
        if (raw.notna() & converted.isna()).any():
            raise ValueError(f"{name} contains non-numeric values.")
        if np.isinf(converted).any() or (converted < 0).any():
            raise ValueError(f"{name} must be finite and non-negative.")
        if name != "TotalCharges" and converted.isna().any():
            raise ValueError(f"{name} cannot be missing.")
        result[name] = converted.astype(float)
    if (result["tenure"] % 1 != 0).any():
        raise ValueError("tenure must be a whole number of months.")
    for name, choices in CATEGORIES.items():
        values = result[name]
        if name == "SeniorCitizen":
            values = values.replace({0: "0", 1: "1"})
        values = values.astype("string").str.strip()
        invalid = ~values.isin(choices)
        if invalid.any():
            raise ValueError(f"{name} must be one of: {', '.join(choices)}.")
        result[name] = values.astype(object)
    no_phone = result["PhoneService"].eq("No")
    if (no_phone != result["MultipleLines"].eq("No phone service")).any():
        raise ValueError("MultipleLines must agree with PhoneService.")
    no_internet = result["InternetService"].eq("No")
    for name in INTERNET_ADDONS:
        if (no_internet != result[name].eq("No internet service")).any():
            raise ValueError(f"{name} must agree with InternetService.")
    return result


def load_dataset(path: Path | str = DATA_PATH) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return X, binary y and unique customer IDs without learning from the dataset."""
    frame = pd.read_csv(path)
    required = {"customerID", "Churn"}
    if not required.issubset(frame):
        raise ValueError("Dataset requires customerID and Churn columns.")
    if frame["customerID"].isna().any() or frame["customerID"].duplicated().any():
        raise ValueError("customerID must be present and unique to avoid customer leakage.")
    labels = frame["Churn"].astype("string").str.strip()
    if not labels.isin(["Yes", "No"]).all():
        raise ValueError("Churn must contain only Yes or No.")
    target = labels.map({"No": 0, "Yes": 1}).astype(int)
    if target.nunique() != 2:
        raise ValueError("Both churn classes are required.")
    return prepare_features(frame), target, frame["customerID"].astype(str)
