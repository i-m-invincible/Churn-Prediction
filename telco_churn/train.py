"""Reproducible model comparison. Run with python -m telco_churn.train."""

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from telco_churn.artifacts import (
    ARTIFACT_FILES,
    HASH_POLICY,
    artifact_sha256,
    package_version,
    sha256,
)
from telco_churn.data import load_dataset
from telco_churn.eda import write_eda
from telco_churn.evaluation import evaluate
from telco_churn.modeling import build_candidates
from telco_churn.schema import (
    ARTIFACT_DIR,
    DATA_PATH,
    FEATURES,
    NUMERIC_FEATURES,
    REPORT_DIR,
    ROOT,
    SEED,
    THRESHOLD,
)


def reference_customer(features: pd.DataFrame) -> dict:
    """Select an actual training row close to training medians and modes."""
    distance = np.zeros(len(features))
    for name in FEATURES:
        if name in NUMERIC_FEATURES:
            values = features[name]
            spread = max(float(values.quantile(0.75) - values.quantile(0.25)), 1.0)
            distance += ((values - values.median()).abs().fillna(spread) / spread).to_numpy()
        else:
            distance += features[name].ne(features[name].mode().iloc[0]).to_numpy()
    return features.iloc[int(distance.argmin())].to_dict()


def comparison_markdown(results: dict) -> str:
    lines = [
        "| Model | CV ROC-AUC (mean ± std) | Test precision | Test recall | "
        "Test F1 | Test ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in results["models"].items():
        cv, test = item["cv"], item["test"]
        label = name + (" **(selected)**" if name == results["selected_model"] else "")
        lines.append(
            f"| {label} | {cv['roc_auc_mean']:.4f} ± {cv['roc_auc_std']:.4f} | "
            f"{test['precision']:.4f} | {test['recall']:.4f} | "
            f"{test['f1']:.4f} | {test['roc_auc']:.4f} |"
        )
    return "\n".join(lines)


def update_readme(results: dict, path: Path) -> None:
    if not path.exists():
        return
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    text = path.read_text()
    if start not in text or end not in text:
        raise ValueError("README requires results markers for automatic metric updates.")
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    path.write_text(before + start + "\n" + comparison_markdown(results) + "\n" + end + after)


def train(
    data_path=DATA_PATH, artifact_dir=ARTIFACT_DIR, report_dir=REPORT_DIR, seed=SEED, folds=5
) -> dict:
    data_path, artifact_dir, report_dir = map(Path, [data_path, artifact_dir, report_dir])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    features, target, ids = load_dataset(data_path)
    train_idx, test_idx = train_test_split(
        np.arange(len(features)), test_size=0.2, stratify=target, random_state=seed
    )
    x_train, x_test = features.iloc[train_idx], features.iloc[test_idx]
    y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
    if set(ids.iloc[train_idx]) & set(ids.iloc[test_idx]):
        raise ValueError("Customer overlap between training and test partitions.")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    candidates = build_candidates(seed)
    scoring = {"roc_auc": "roc_auc", "precision": "precision", "recall": "recall", "f1": "f1"}
    results = {
        "dataset_sha256": sha256(data_path),
        "random_seed": seed,
        "decision_threshold": THRESHOLD,
        "selection_rule": "Maximum mean training-only stratified CV ROC-AUC; name breaks ties.",
        "cv_folds": folds,
        "train_customers": len(x_train),
        "test_customers": len(x_test),
        "train_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
        "models": {},
    }
    # Finish all model selection before computing any test-set metrics.
    for name, pipeline in candidates.items():
        print(f"Cross-validating {name} ({folds} folds)...", flush=True)
        scores = cross_validate(
            pipeline, x_train, y_train, cv=splitter, scoring=scoring, n_jobs=1, error_score="raise"
        )
        cv = {}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            cv[f"{metric}_mean"] = float(values.mean())
            cv[f"{metric}_std"] = float(values.std())
            cv[f"{metric}_folds"] = values.tolist()
        results["models"][name] = {"cv": cv, "parameters": pipeline["model"].get_params()}
    selected = sorted(candidates, key=lambda n: (-results["models"][n]["cv"]["roc_auc_mean"], n))[0]
    results["selected_model"] = selected
    print(f"Selected by CV: {selected}. Evaluating frozen candidates on held-out test data.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    test_predictions = pd.DataFrame({"customerID": ids.iloc[test_idx], "actual_churn": y_test})
    for name, pipeline in candidates.items():
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        results["models"][name]["test"] = evaluate(y_test, probabilities, THRESHOLD)
        test_predictions[name] = probabilities
        RocCurveDisplay.from_predictions(y_test, probabilities, name=name, ax=axes[0])
        CalibrationDisplay.from_predictions(y_test, probabilities, name=name, n_bins=8, ax=axes[1])
    figures = report_dir / "figures"
    figures.mkdir(exist_ok=True)
    axes[0].set_title("Held-out ROC curves")
    axes[1].set_title("Held-out reliability curves")
    fig.savefig(figures / "evaluation.png", dpi=150)
    plt.close(fig)
    pipeline = candidates[selected]
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        test_predictions[selected].ge(THRESHOLD).astype(int),
        labels=[0, 1],
        display_labels=["Stay", "Churn"],
        ax=ax,
        colorbar=False,
        cmap="Blues",
    )
    ax.set_title(f"{selected}: held-out test data")
    fig.savefig(figures / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    importance = permutation_importance(
        pipeline, x_test, y_test, scoring="roc_auc", n_repeats=5, random_state=seed, n_jobs=1
    )
    pd.DataFrame(
        {
            "feature": FEATURES,
            "roc_auc_drop": importance.importances_mean,
            "std": importance.importances_std,
        }
    ).sort_values("roc_auc_drop", ascending=False).to_csv(
        report_dir / "feature_importance.csv", index=False
    )
    write_eda(x_train, y_train, figures)
    # The model remains fitted to the development partition so its published score
    # describes the exact model served by the dashboard.
    joblib.dump(pipeline, artifact_dir / "churn_pipeline.joblib", compress=3)
    joblib.dump(pipeline["preprocess"], artifact_dir / "preprocessor.joblib", compress=3)
    joblib.dump(pipeline["model"], artifact_dir / "model.joblib", compress=3)
    (artifact_dir / "reference_customer.json").write_text(
        json.dumps(reference_customer(x_train), indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "xgboost",
        "joblib",
        "streamlit",
        "matplotlib",
        "seaborn",
    ]
    metadata = {
        "artifact_schema_version": 1,
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "selected_model": selected,
        "decision_threshold": THRESHOLD,
        "risk_bands": {"Low": "p < 0.30", "Moderate": "0.30 <= p < 0.60", "High": "p >= 0.60"},
        "features": FEATURES,
        "encoded_features": pipeline["preprocess"].get_feature_names_out().tolist(),
        "dataset_sha256": results["dataset_sha256"],
        "python_version": platform.python_version(),
        "versions": {p: package_version(p) for p in packages},
        "train_customers": len(x_train),
        "test_customers": len(x_test),
        "seed": seed,
        "cv_folds": folds,
        "selection_rule": results["selection_rule"],
        "selected_test_metrics": results["models"][selected]["test"],
        "artifact_hash_policy": HASH_POLICY,
        "artifact_sha256": {name: artifact_sha256(artifact_dir / name) for name in ARTIFACT_FILES},
    }
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (report_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    (report_dir / "model_comparison.md").write_text(comparison_markdown(results) + "\n")
    manifest = pd.DataFrame({"customerID": ids, "partition": "train"})
    manifest.loc[test_idx, "partition"] = "test"
    manifest.to_csv(report_dir / "split_manifest.csv", index=False)
    test_predictions.to_csv(report_dir / "test_predictions.csv", index=False)
    if report_dir.resolve() == REPORT_DIR.resolve():
        update_readme(results, ROOT / "README.md")
    print(comparison_markdown(results), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--reports", type=Path, default=REPORT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    train(args.data, args.artifacts, args.reports, args.seed, args.folds)


if __name__ == "__main__":
    main()
