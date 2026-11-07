# Customer Churn Prediction | Churn Atlas

An explainable customer churn dashboard: assess one customer, score a CSV, and inspect the evidence behind the model.

[GitHub repository](https://github.com/i-m-invincible/Churn-Prediction) · [Deployment guide](docs/deployment.md)

## Explore the application

| Workspace | What you can do |
|---|---|
| **Predict Churn** | Try example profiles, estimate churn probability, inspect risk factors, and download CSV batch predictions |
| **Explore Data** | Compare customer groups, churn rates, tenure and correlations using training data only |
| **Model Performance** | Inspect real model comparisons, ROC and reliability curves, confusion matrices and feature sensitivity |

Built with Python 3.12, Streamlit, Pandas, NumPy, scikit-learn, XGBoost, Altair, Matplotlib and Seaborn.

## Model results

The model is selected by the highest mean **five-fold training-only CV ROC-AUC**. The table's test metrics use the same **1,409 held-out customers**, Churn = 1, and a fixed **0.50 classification threshold**.

<!-- RESULTS:START -->
| Model | CV ROC-AUC (mean ± std) | Test precision | Test recall | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8461 ± 0.0125 | 0.6572 | 0.5588 | 0.6040 | 0.8419 |
| Random Forest | 0.8475 ± 0.0090 | 0.6786 | 0.5080 | 0.5810 | 0.8439 |
| XGBoost **(selected)** | 0.8477 ± 0.0106 | 0.6689 | 0.5294 | 0.5910 | 0.8469 |
<!-- RESULTS:END -->

The CV differences are small; this is not evidence of statistically significant superiority. Logistic Regression has the highest test F1 at the fixed threshold. [Full metrics and parameters](reports/metrics.json) · [Test predictions](reports/test_predictions.csv).

## How it works

1. Validate 19 raw predictors, exclude customer ID and target, and convert blank total charges to missing values.
2. Create a stratified 80/20 split: **5,634 training / 1,409 test customers**, seed 42.
3. Within every CV fold, fit numeric median imputation and scaling plus categorical imputation and one-hot encoding.
4. Select by training CV, fit on the training partition, then evaluate the frozen candidates on the test partition.
5. Serve the exact evaluated model from verified saved artifacts. Application startup performs inference, **not training**.

**Risk bands:** low <30%, moderate 30–60%, high ≥60%. These are illustrative policy bands, separate from the 50% class threshold.

**Explanations:** replace one field group at a time with values from a representative actual training customer and measure the probability difference. Related service fields move together. These sensitivities are non-additive and do not establish causes. Global permutation importance is inspected after selection.

## Run locally

Use **Python 3.12** to match deployment and the saved model environment.

```bash
git clone https://github.com/i-m-invincible/Churn-Prediction.git
cd Churn-Prediction
```

Create and activate an environment:

| Platform | Create | Activate |
|---|---|---|
| Windows PowerShell | `py -3.12 -m venv .venv` | `.venv\Scripts\Activate.ps1` |
| macOS / Linux | `python3.12 -m venv .venv` | `source .venv/bin/activate` |

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts/verify_artifacts.py
streamlit run app.py
```

The dataset and trained artifacts are included. No API keys or network access are needed to run locally. Batch uploads accept UTF-8 CSV files up to 5 MB / 10,000 customers; an [example CSV](examples/customer_batch.csv) is included.

To reproduce training and refresh the generated reports and this results table:

```bash
python -m telco_churn.train
```

## Project structure

| Path | Purpose |
|---|---|
| `app.py` | Streamlit entrypoint |
| `telco_churn/` | Dashboard, validation, preprocessing, training, inference and integrity checks |
| `artifacts/` | Verified pipeline, estimator, preprocessor, reference profile and metadata |
| `data/`, `examples/` | Pinned source dataset and batch example |
| `reports/` | Metrics, split membership, predictions, importance and figures |
| `tests/`, `.github/workflows/` | Tests and Windows/Linux CI |
| `docs/`, `third_party/` | Deployment guidance and retained notices |

## Data and limitations

The [IBM Telco Customer Churn dataset](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/d5371f5d83a446ad5673cbcca3b814b926491f8a/data/Telco-Customer-Churn.csv) contains **7,043 customers**, including **1,869 churners**, and 11 blank total-charge entries. [Data provenance and checksum](data/README.md).

This historical sample does not establish performance for a current telecom provider. Probabilities need calibration checks on recent data; production use also needs cost-based thresholds, subgroup evaluation and drift monitoring. Future work includes nested tuning, calibration and time-based validation.

## Checks and deployment

```bash
python -m pip install -r requirements-dev.txt
ruff check .
python scripts/verify_artifacts.py
pytest -q
```

Deploy on **Streamlit Community Cloud** using repository `i-m-invincible/Churn-Prediction`, branch `main`, entrypoint `app.py`, and **Python 3.12 in Advanced settings**. No Docker, paid APIs, secrets or additional hosting are required. [Deployment and troubleshooting](docs/deployment.md).

Linux x86_64 uses the matching CPU-only XGBoost wheel. Caches are bounded, uploads are capped, and internal exceptions are hidden from visitors. Binary artifacts are verified exactly; reference JSON verification tolerates CRLF-to-LF transport changes while rejecting other changes.

## Author and acknowledgements

**Bishal Dutta** · [i-m-invincible](https://github.com/i-m-invincible)

Developed with AI assistance for implementation, debugging and documentation.

This project was inspired by the Customer Churn Prediction project developed by Ibrahim Khan (ibrahimkhan-data). The implementation in this repository was independently developed, with inspiration drawn from the original project's concept and approach.

[Original project](https://github.com/ibrahimkhan-data/Customer-Churn-Prediction) · [Third-party notices](THIRD_PARTY_NOTICES.md) · [MIT license for application code](LICENSE)
