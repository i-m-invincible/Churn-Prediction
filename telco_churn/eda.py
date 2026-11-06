"""Exploratory plots restricted to the development partition."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from telco_churn.schema import NUMERIC_FEATURES


def write_eda(features: pd.DataFrame, target: pd.Series, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    frame = features.assign(Churn=target)
    summary = {
        "partition": "training only",
        "customers": len(frame),
        "churn_count": int(target.sum()),
        "churn_rate": float(target.mean()),
        "missing": features.isna().sum().astype(int).to_dict(),
        "numeric_summary": features[NUMERIC_FEATURES].describe().to_dict(),
        "contract_churn_rates": frame.groupby("Contract")["Churn"].mean().to_dict(),
    }
    (directory.parent / "eda_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    sns.set_theme(style="whitegrid", palette="colorblind")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    sns.countplot(data=frame, x="Churn", ax=axes[0], color="#167D8D")
    axes[0].set_xticks([0, 1], ["Stayed", "Churned"])
    axes[0].set_title("Target balance")
    sns.boxplot(data=frame, x="Churn", y="tenure", ax=axes[1], color="#70B7C0")
    axes[1].set_xticks([0, 1], ["Stayed", "Churned"])
    axes[1].set_title("Tenure and churn")
    rates = frame.groupby("Contract")["Churn"].mean().sort_values()
    axes[2].barh(rates.index, rates.values, color="#167D8D")
    axes[2].set_xlabel("Observed churn fraction")
    axes[2].set_title("Contract groups")
    fig.suptitle("Development data only — descriptive associations")
    fig.savefig(directory / "eda_overview.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    sns.heatmap(
        frame[NUMERIC_FEATURES + ["Churn"]].corr(),
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Numeric correlations: training partition")
    fig.savefig(directory / "correlations.png", dpi=150)
    plt.close(fig)
    return summary
