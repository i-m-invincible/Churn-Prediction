import numpy as np
import pytest

from telco_churn.data import load_dataset, prepare_features
from telco_churn.schema import FEATURES, INTERNET_ADDONS


def test_dataset_contract(dataset):
    features, target, ids = dataset
    assert len(features) == len(target) == len(ids) == 7043
    assert ids.is_unique
    assert list(features) == FEATURES
    assert "customerID" not in features and "Churn" not in features
    assert set(target) == {0, 1}
    assert features["TotalCharges"].isna().sum() == 11


def test_numeric_and_blank_cleaning_is_stateless(dataset):
    sample = dataset[0].head(1).copy()
    sample["TotalCharges"] = "  "
    sample["tenure"] = "12"
    result = prepare_features(sample)
    assert np.isnan(result["TotalCharges"].iloc[0])
    assert result["tenure"].iloc[0] == 12
    assert sample["TotalCharges"].iloc[0] == "  "


@pytest.mark.parametrize(
    "field,value",
    [
        ("tenure", -1),
        ("tenure", 1.5),
        ("tenure", None),
        ("MonthlyCharges", np.inf),
        ("MonthlyCharges", "oops"),
        ("TotalCharges", -100),
        ("gender", "unknown"),
        ("Contract", None),
        ("SeniorCitizen", 2),
    ],
)
def test_invalid_inputs_are_rejected(dataset, field, value):
    sample = dataset[0].head(1).copy().astype(object)
    sample[field] = value
    with pytest.raises(ValueError, match=field):
        prepare_features(sample)


def test_missing_columns_and_empty_inputs_are_rejected(dataset):
    with pytest.raises(ValueError, match="Contract"):
        prepare_features(dataset[0].drop(columns="Contract"))
    with pytest.raises(ValueError, match="At least one"):
        prepare_features(dataset[0].head(0))


def test_service_validation(dataset):
    sample = dataset[0].head(1).copy()
    sample["PhoneService"] = "Yes"
    sample["MultipleLines"] = "No phone service"
    with pytest.raises(ValueError, match="MultipleLines"):
        prepare_features(sample)
    sample["MultipleLines"] = "Yes"
    sample["InternetService"] = "No"
    with pytest.raises(ValueError, match="InternetService"):
        prepare_features(sample)
    for name in INTERNET_ADDONS:
        sample[name] = "No internet service"
    assert len(prepare_features(sample)) == 1


def test_duplicate_customers_cannot_cross_partitions(tmp_path, dataset):
    features, target, ids = dataset
    sample = features.head(10).assign(
        customerID=ids.head(10), Churn=target.head(10).map({0: "No", 1: "Yes"})
    )
    sample.loc[1, "customerID"] = sample.loc[0, "customerID"]
    path = tmp_path / "duplicates.csv"
    sample.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_dataset(path)


def test_unknown_target_is_rejected(tmp_path, dataset):
    frame = dataset[0].head(10).assign(customerID=dataset[2].head(10), Churn="Maybe")
    path = tmp_path / "bad_labels.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Churn"):
        load_dataset(path)


def test_numeric_and_string_senior_status_match(dataset):
    sample = dataset[0].head(2).copy()
    sample["SeniorCitizen"] = [0.0, 1.0]
    assert prepare_features(sample)["SeniorCitizen"].tolist() == ["0", "1"]
