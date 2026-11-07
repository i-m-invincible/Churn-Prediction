"""Verified artifact loading, batch scoring and model-based local comparisons."""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from telco_churn.artifacts import ARTIFACT_FILES, artifact_sha256, package_version
from telco_churn.artifacts import sha256 as sha256
from telco_churn.data import prepare_features
from telco_churn.schema import ARTIFACT_DIR, EXPLANATION_GROUPS


@dataclass
class ModelBundle:
    pipeline: Pipeline
    metadata: dict
    reference: pd.DataFrame


def load_bundle(directory: Path | str = ARTIFACT_DIR) -> ModelBundle:
    """Load only the repository's trusted files; never accept uploaded joblib files."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("artifact_schema_version") != 1:
        raise ValueError("Unsupported artifact schema. Regenerate artifacts.")
    for package in ["scikit-learn", "xgboost"]:
        if package_version(package) != metadata["versions"][package]:
            raise ValueError(f"{package} version differs from training. Install requirements.txt.")
    for filename in ARTIFACT_FILES:
        path = directory / filename
        if artifact_sha256(path) != metadata["artifact_sha256"][filename]:
            raise ValueError(f"Artifact checksum mismatch: {filename}. Regenerate artifacts.")
    pipeline = joblib.load(directory / "churn_pipeline.joblib")
    reference = prepare_features(
        pd.DataFrame(
            [json.loads((directory / "reference_customer.json").read_text(encoding="utf-8"))]
        )
    )
    if metadata["features"] != pipeline.feature_names_in_.tolist():
        raise ValueError("Artifact input schema differs from metadata.")
    encoded = pipeline["preprocess"].get_feature_names_out().tolist()
    if encoded != metadata["encoded_features"]:
        raise ValueError("Artifact preprocessing schema differs from metadata.")
    return ModelBundle(pipeline=pipeline, metadata=metadata, reference=reference)


def risk_band(probability: float) -> str:
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")
    return "High" if probability >= 0.6 else "Moderate" if probability >= 0.3 else "Low"


def predict_customers(bundle: ModelBundle, frame: pd.DataFrame) -> pd.DataFrame:
    features = prepare_features(frame)
    probabilities = bundle.pipeline.predict_proba(features)[:, 1]
    return pd.DataFrame(
        {
            "churn_probability": probabilities,
            "predicted_churn": probabilities >= bundle.metadata["decision_threshold"],
            "risk_band": [risk_band(float(p)) for p in probabilities],
        },
        index=frame.index,
    )


def explain_customer(bundle: ModelBundle, frame: pd.DataFrame) -> pd.DataFrame:
    """One group replaced at a time, relative to a fixed actual training customer.

    The signed probability differences are sensitivities, not additive attributions
    or causal effects. Related service fields are replaced together for consistency.
    """
    features = prepare_features(frame)
    if len(features) != 1:
        raise ValueError("Local explanations require exactly one customer.")
    current = features.iloc[[0]].reset_index(drop=True)
    variants = []
    for columns in EXPLANATION_GROUPS.values():
        alternate = current.copy()
        for name in columns:
            alternate[name] = bundle.reference.iloc[0][name]
        variants.append(alternate)
    probabilities = bundle.pipeline.predict_proba(
        pd.concat([current, *variants], ignore_index=True)
    )[:, 1]
    explanations = []
    for i, (label, columns) in enumerate(EXPLANATION_GROUPS.items(), 1):
        explanations.append(
            {
                "factor": label,
                "customer_value": "; ".join(f"{c}: {current.iloc[0][c]}" for c in columns),
                "reference_value": "; ".join(
                    f"{c}: {bundle.reference.iloc[0][c]}" for c in columns
                ),
                "probability_difference_pp": float(100 * (probabilities[0] - probabilities[i])),
                "replacement_probability": float(probabilities[i]),
            }
        )
    return (
        pd.DataFrame(explanations)
        .sort_values("probability_difference_pp", key=abs, ascending=False)
        .reset_index(drop=True)
    )
