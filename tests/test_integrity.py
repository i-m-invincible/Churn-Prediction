import json
import shutil
from unittest.mock import Mock

import numpy as np
import pytest

from telco_churn.artifacts import ARTIFACT_FILES, artifact_sha256, sha256
from telco_churn.inference import load_bundle, predict_customers
from telco_churn.schema import ARTIFACT_DIR


@pytest.fixture
def copied_artifacts(tmp_path):
    shutil.copytree(ARTIFACT_DIR, tmp_path / "artifacts")
    return tmp_path / "artifacts"


def test_crlf_reference_preserves_verification_and_predictions(copied_artifacts, bundle, dataset):
    path = copied_artifacts / "reference_customer.json"
    original = path.read_bytes()
    expected = artifact_sha256(path)
    path.write_bytes(original.replace(b"\n", b"\r\n"))
    assert sha256(path) != expected
    assert artifact_sha256(path) == expected
    loaded = load_bundle(copied_artifacts)
    np.testing.assert_array_equal(
        predict_customers(loaded, dataset[0].head(8)).churn_probability,
        predict_customers(bundle, dataset[0].head(8)).churn_probability,
    )


@pytest.mark.parametrize("filename", ARTIFACT_FILES)
def test_changed_artifacts_are_rejected_before_deserialization(
    copied_artifacts, filename, monkeypatch
):
    path = copied_artifacts / filename
    if filename.endswith("json"):
        content = path.read_bytes().replace(b'"tenure": 17.0', b'"tenure": 18.0')
        assert content != path.read_bytes()
        path.write_bytes(content)
    else:
        path.write_bytes(path.read_bytes() + b"corrupted")
    loader = Mock(side_effect=AssertionError("No artifact may be loaded before verification"))
    monkeypatch.setattr("telco_churn.inference.joblib.load", loader)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_bundle(copied_artifacts)
    loader.assert_not_called()


def test_reference_formatting_changes_are_not_silently_accepted(copied_artifacts):
    path = copied_artifacts / "reference_customer.json"
    path.write_text(json.dumps(json.loads(path.read_text()), indent=4), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_bundle(copied_artifacts)


def test_preprocessing_metadata_mismatch_is_rejected(copied_artifacts):
    path = copied_artifacts / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["encoded_features"][0] = "unexpected_feature"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="preprocessing schema"):
        load_bundle(copied_artifacts)
