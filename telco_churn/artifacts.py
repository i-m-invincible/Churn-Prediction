"""Shared, deterministic artifact integrity rules for training and serving."""

import hashlib
import importlib.metadata
from pathlib import Path

ARTIFACT_FILES = (
    "churn_pipeline.joblib",
    "preprocessor.joblib",
    "model.joblib",
    "reference_customer.json",
)
HASH_POLICY = "sha256; reference JSON CRLF normalized to LF; model binaries exact"


def sha256(path: Path) -> str:
    """Hash exact file bytes, as required for models and the source dataset."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_sha256(path: Path) -> str:
    """Allow only transport-level CRLF differences in the reference JSON.

    Every other byte, including whitespace and JSON values, remains significant.
    Binary model files are never normalized or deserialized before verification.
    """
    if path.name != "reference_customer.json":
        return sha256(path)
    content = path.read_bytes()
    content.decode("utf-8", errors="strict")
    return hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()


def package_version(package: str) -> str:
    """CPU and full XGBoost wheels expose the same versioned Python API."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        if package != "xgboost":
            raise
        return importlib.metadata.version("xgboost-cpu")
