import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from telco_churn.modeling import build_preprocessor
from telco_churn.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SEED


def test_transformers_never_fit_on_validation_values(dataset):
    train = dataset[0].head(30).copy()
    validation = dataset[0].iloc[[40]].copy()
    train["Contract"] = "train-only-category"
    validation["Contract"] = "validation-only-category"
    validation["MonthlyCharges"] = 1_000_000
    preprocessor = build_preprocessor().fit(train)
    numeric = preprocessor.named_transformers_["numeric"]
    encoder = preprocessor.named_transformers_["categorical"]["encode"]
    contract_index = CATEGORICAL_FEATURES.index("Contract")
    assert encoder.categories_[contract_index].tolist() == ["train-only-category"]
    np.testing.assert_allclose(numeric["scale"].mean_, train[NUMERIC_FEATURES].mean())
    transformed = preprocessor.transform(validation)
    assert np.isfinite(transformed).all()
    assert transformed.shape[1] == len(preprocessor.get_feature_names_out())


def test_saved_preprocessor_statistics_match_training_only(bundle, dataset):
    features, target, _ = dataset
    train_idx, test_idx = train_test_split(
        np.arange(len(features)), test_size=0.2, stratify=target, random_state=SEED
    )
    numeric = bundle.pipeline["preprocess"].named_transformers_["numeric"]
    medians = features.iloc[train_idx][NUMERIC_FEATURES].median().to_numpy()
    np.testing.assert_allclose(numeric["impute"].statistics_, medians)
    filled = features.iloc[train_idx][NUMERIC_FEATURES].fillna(
        features.iloc[train_idx][NUMERIC_FEATURES].median()
    )
    np.testing.assert_allclose(numeric["scale"].mean_, filled.mean().to_numpy())
    assert len(set(train_idx) & set(test_idx)) == 0


def test_split_manifest_reproduces_stratified_partition(dataset):
    from telco_churn.schema import REPORT_DIR

    features, target, ids = dataset
    _, test_idx = train_test_split(
        np.arange(len(features)), test_size=0.2, stratify=target, random_state=SEED
    )
    manifest = pd.read_csv(REPORT_DIR / "split_manifest.csv")
    assert len(manifest) == len(features)
    assert manifest.customerID.is_unique
    assert set(manifest.loc[manifest.partition.eq("test"), "customerID"]) == set(ids.iloc[test_idx])
