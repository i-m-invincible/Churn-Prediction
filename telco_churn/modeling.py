"""Fresh preprocessing pipelines for each estimator and each validation fold."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from telco_churn.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SEED


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_candidates(seed: int = SEED) -> dict[str, Pipeline]:
    """Fixed baseline settings, declared before looking at held-out results."""
    estimators = {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=8,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=seed,
            n_jobs=2,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=350,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=5,
            reg_lambda=5.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=2,
        ),
    }
    return {
        name: Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        for name, estimator in estimators.items()
    }
