import json
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from telco_churn.inference import (
    explain_customer,
    load_bundle,
    predict_customers,
    risk_band,
    sha256,
)
from telco_churn.schema import ARTIFACT_DIR, DATA_PATH, EXPLANATION_GROUPS, REPORT_DIR, ROOT
from telco_churn.train import comparison_markdown


def test_all_artifacts_agree(bundle, dataset):
    sample = dataset[0].head(20)
    scored = predict_customers(bundle, sample)
    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    preprocessor = joblib.load(ARTIFACT_DIR / "preprocessor.joblib")
    np.testing.assert_allclose(
        model.predict_proba(preprocessor.transform(sample))[:, 1],
        scored.churn_probability,
        rtol=1e-6,
    )
    assert scored.churn_probability.between(0, 1).all()
    assert set(scored.risk_band) <= {"Low", "Moderate", "High"}


def test_batch_and_single_predictions_agree(bundle, dataset):
    sample = dataset[0].head(4).copy()
    sample.index = [3, 20, 9, 42]
    sample.loc[20, "TotalCharges"] = np.nan
    scored = predict_customers(bundle, sample)
    singles = [
        predict_customers(bundle, sample.iloc[[i]]).churn_probability.iloc[0] for i in range(4)
    ]
    np.testing.assert_allclose(scored.churn_probability, singles)
    assert scored.index.tolist() == sample.index.tolist()


def test_corrupt_artifact_is_rejected_before_loading(tmp_path):
    for path in ARTIFACT_DIR.iterdir():
        if path.is_file():
            shutil.copy(path, tmp_path / path.name)
    with (tmp_path / "churn_pipeline.joblib").open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_bundle(tmp_path)


@pytest.mark.parametrize(
    "p,expected",
    [
        (0, "Low"),
        (0.2999, "Low"),
        (0.3, "Moderate"),
        (0.5999, "Moderate"),
        (0.6, "High"),
        (1, "High"),
    ],
)
def test_risk_boundaries(p, expected):
    assert risk_band(p) == expected


@pytest.mark.parametrize("p", [-0.1, 1.1, np.nan, np.inf])
def test_risk_rejects_invalid_probability(p):
    with pytest.raises(ValueError):
        risk_band(p)


def test_explanations_are_actual_model_probability_comparisons(bundle, dataset):
    sample = dataset[0].iloc[[7]].reset_index(drop=True)
    explanation = explain_customer(bundle, sample).set_index("factor")
    original_p = bundle.pipeline.predict_proba(sample)[0, 1]
    for label, columns in EXPLANATION_GROUPS.items():
        alternative = sample.copy()
        for name in columns:
            alternative[name] = bundle.reference.iloc[0][name]
        alternative_p = predict_customers(bundle, alternative).churn_probability.iloc[0]
        assert explanation.loc[label, "probability_difference_pp"] == pytest.approx(
            100 * (original_p - alternative_p), abs=1e-5
        )


def test_published_metrics_and_selection_are_reproducible(bundle, dataset):
    metrics = json.loads((REPORT_DIR / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(REPORT_DIR / "test_predictions.csv")
    features, target, ids = dataset
    test_ids = predictions.customerID.tolist()
    indexed = features.set_index(ids)
    target_indexed = pd.Series(target.to_numpy(), index=ids)
    np.testing.assert_array_equal(target_indexed.loc[test_ids], predictions.actual_churn)
    selected = max(metrics["models"], key=lambda n: metrics["models"][n]["cv"]["roc_auc_mean"])
    assert selected == metrics["selected_model"] == bundle.metadata["selected_model"]
    live = predict_customers(bundle, indexed.loc[test_ids]).churn_probability
    np.testing.assert_allclose(live, predictions[selected], atol=1e-7)
    for name, item in metrics["models"].items():
        y, p = predictions.actual_churn, predictions[name]
        actual = {
            "precision": precision_score(y, p >= 0.5),
            "recall": recall_score(y, p >= 0.5),
            "f1": f1_score(y, p >= 0.5),
            "roc_auc": roc_auc_score(y, p),
        }
        for metric, value in actual.items():
            assert item["test"][metric] == pytest.approx(value, abs=1e-8)
        assert len(item["cv"]["roc_auc_folds"]) == 5
    assert comparison_markdown(metrics) in (ROOT / "README.md").read_text(encoding="utf-8")
    assert metrics["dataset_sha256"] == sha256(DATA_PATH)
    assert metrics["train_customers"] + metrics["test_customers"] == len(features)


def test_reference_customer_really_exists_in_training(bundle, dataset):
    manifest = pd.read_csv(REPORT_DIR / "split_manifest.csv")
    features, _, ids = dataset
    train = features.loc[ids.isin(manifest.loc[manifest.partition.eq("train"), "customerID"])]
    reference = bundle.reference.iloc[0]
    assert train.eq(reference).all(axis=1).any()
