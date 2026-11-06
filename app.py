"""Streamlit entry point: streamlit run app.py."""

import json

import pandas as pd
import streamlit as st

from telco_churn.data import load_dataset
from telco_churn.inference import explain_customer, load_bundle, predict_customers
from telco_churn.schema import ARTIFACT_DIR, CATEGORIES, INTERNET_ADDONS, REPORT_DIR

st.set_page_config(page_title="Churn Atlas · Bishal Dutta", page_icon="📊", layout="wide")


@st.cache_resource
def resources(metadata_revision: str):
    return load_bundle()


@st.cache_data
def development_data():
    features, target, ids = load_dataset()
    manifest = pd.read_csv(REPORT_DIR / "split_manifest.csv")
    train_ids = set(manifest.loc[manifest["partition"].eq("train"), "customerID"])
    mask = ids.isin(train_ids)
    return features.loc[mask].assign(Churn=target.loc[mask])


st.title("Churn Atlas")
st.caption("Customer retention insights · Built by Bishal Dutta")
try:
    bundle = resources((ARTIFACT_DIR / "metadata.json").read_text())
    metrics = json.loads((REPORT_DIR / "metrics.json").read_text())
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Cannot load the project artifacts: {exc}")
    st.info("Install requirements.txt, then run: python -m telco_churn.train")
    st.stop()

page = st.sidebar.radio("Workspace", ["Predict churn", "Explore data", "Model performance"])
st.sidebar.divider()
st.sidebar.markdown(f"**Selected model**  \n{bundle.metadata['selected_model']}")
st.sidebar.caption("Selected with training-only cross-validation. Test data is held out.")
st.sidebar.markdown("[Source code](https://github.com/i-m-invincible/Churn-Prediction)")

if page == "Predict churn":
    st.subheader("Assess a customer's churn risk")
    st.write("Enter account and service details. The score estimates the chance of Churn = Yes.")
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        with st.container(border=True):
            # Parent service choices are outside the form so dependent fields update immediately.
            a, b = st.columns(2)
            phone = a.selectbox("Phone service", ["Yes", "No"])
            internet = b.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            with st.form("customer_form"):
                raw = {"PhoneService": phone, "InternetService": internet}
                st.markdown("**Account and billing**")
                a, b = st.columns(2)
                raw["tenure"] = a.number_input("Tenure (months)", 0, 120, 12)
                raw["Contract"] = b.selectbox("Contract", CATEGORIES["Contract"])
                raw["MonthlyCharges"] = a.number_input("Monthly charges ($)", 0.0, 1000.0, 70.0)
                raw["TotalCharges"] = b.number_input("Total charges ($)", 0.0, 120000.0, 840.0)
                unknown_total = st.checkbox("Total charges are unavailable")
                raw["PaymentMethod"] = st.selectbox("Payment method", CATEGORIES["PaymentMethod"])
                raw["PaperlessBilling"] = st.selectbox("Paperless billing", ["Yes", "No"])
                st.markdown("**Customer details**")
                a, b = st.columns(2)
                raw["gender"] = a.selectbox("Gender", CATEGORIES["gender"])
                raw["SeniorCitizen"] = (
                    "1" if b.selectbox("Senior citizen", ["No", "Yes"]) == "Yes" else "0"
                )
                raw["Partner"] = a.selectbox("Partner", ["No", "Yes"])
                raw["Dependents"] = b.selectbox("Dependents", ["No", "Yes"])
                st.markdown("**Service options**")
                raw["MultipleLines"] = st.selectbox(
                    "Multiple lines",
                    ["No", "Yes"] if phone == "Yes" else ["No phone service"],
                    key=f"lines_{phone}",
                )
                a, b = st.columns(2)
                labels = [
                    "Online security",
                    "Online backup",
                    "Device protection",
                    "Tech support",
                    "Streaming TV",
                    "Streaming movies",
                ]
                for i, (name, label) in enumerate(zip(INTERNET_ADDONS, labels, strict=True)):
                    raw[name] = (a if i % 2 == 0 else b).selectbox(
                        label,
                        ["No", "Yes"] if internet != "No" else ["No internet service"],
                        key=f"{name}_{internet}",
                    )
                submitted = st.form_submit_button(
                    "Estimate churn risk", type="primary", use_container_width=True
                )
            if submitted:
                if unknown_total:
                    raw["TotalCharges"] = None
                try:
                    frame = pd.DataFrame([raw])
                    result = predict_customers(bundle, frame).iloc[0].to_dict()
                    factors = explain_customer(bundle, frame)
                    st.session_state["assessment"] = (result, factors, raw)
                except ValueError as exc:
                    st.error(str(exc))
    with right:
        with st.container(border=True):
            st.markdown("### Risk assessment")
            if "assessment" not in st.session_state:
                st.info("Submit a customer profile to see the model's estimate and risk factors.")
            else:
                result, factors, snapshot = st.session_state["assessment"]
                p = float(result["churn_probability"])
                st.metric("Estimated churn probability", f"{p:.1%}")
                st.progress(p)
                st.markdown(f"**{result['risk_band']} risk**")
                st.write("Predicted class: " + ("Churn" if result["predicted_churn"] else "Stay"))
                st.caption(
                    "Class threshold: 50%. Risk bands: low <30%, moderate 30–60%, high ≥60%."
                )
                st.markdown("#### Factors affecting this estimate")
                meaningful = factors.loc[factors["probability_difference_pp"].abs().gt(0.01)].head(
                    6
                )
                st.dataframe(
                    meaningful[["factor", "probability_difference_pp"]].rename(
                        columns={
                            "factor": "Factor",
                            "probability_difference_pp": "Difference (percentage points)",
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption(
                    "Each difference is current probability minus the probability after replacing "
                    "that factor with the reference customer's value. Positive values mean higher "
                    "estimated risk. These comparisons do not add up and do not establish causes."
                )
                with st.expander("Compare values with the reference customer"):
                    st.dataframe(factors, hide_index=True)
                    st.write("Reference profile", bundle.reference.iloc[0].to_dict())
                with st.expander("Profile used for this result"):
                    st.json(snapshot)
                st.caption(
                    "This result refers to the last submitted profile. Submit again after edits."
                )
            st.divider()
            st.caption(
                "An educational estimate from a historical sample dataset. Risk bands are "
                "illustrative policies; production use needs recent data, calibration "
                "and fairness checks."
            )
    with st.expander("Score multiple customers from a CSV"):
        st.write(
            "Use the dataset's 19 raw feature names. customerID and Churn are optional and ignored."
        )
        uploaded = st.file_uploader("Customer CSV", type=["csv"])
        if uploaded is not None:
            try:
                if uploaded.size > 5_000_000:
                    raise ValueError("Please use a CSV smaller than 5 MB.")
                frame = pd.read_csv(uploaded)
                if len(frame) > 10_000:
                    raise ValueError("Use at most 10,000 customers per batch.")
                output = predict_customers(bundle, frame)
                if "customerID" in frame:
                    output.insert(0, "customerID", frame["customerID"])
                st.dataframe(output, hide_index=True)
                st.download_button(
                    "Download predictions",
                    output.to_csv(index=False),
                    "churn_predictions.csv",
                    "text/csv",
                )
            except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                st.error(str(exc))

elif page == "Explore data":
    st.subheader("Understand the development data")
    frame = development_data()
    a, b, c = st.columns(3)
    a.metric("Training customers", f"{len(frame):,}")
    b.metric("Observed churn rate", f"{frame['Churn'].mean():.1%}")
    c.metric("Missing total charges", str(frame["TotalCharges"].isna().sum()))
    st.caption(
        "All exploratory views use the training partition. The test partition stays out of EDA."
    )
    st.image(str(REPORT_DIR / "figures" / "eda_overview.png"))
    feature = st.selectbox(
        "Compare customer groups",
        ["Contract", "InternetService", "PaymentMethod", "SeniorCitizen", "TechSupport"],
    )
    rates = frame.groupby(feature, observed=True)["Churn"].agg(["mean", "count"])
    st.bar_chart(rates["mean"].rename("Observed churn fraction"))
    st.dataframe(rates.rename(columns={"mean": "Churn fraction", "count": "Customers"}))
    st.image(str(REPORT_DIR / "figures" / "correlations.png"))
    st.caption("Correlation describes association within this sample; it does not show causation.")

else:
    st.subheader("Model comparison and evaluation")
    st.write(f"Selected model: **{metrics['selected_model']}**")
    st.caption(
        f"{metrics['train_customers']:,} training / "
        f"{metrics['test_customers']:,} held-out test customers. "
        f"{metrics['cv_folds']}-fold stratified CV, random seed {metrics['random_seed']}. "
        "Selection uses mean training-only CV ROC-AUC, before test evaluation."
    )
    rows = []
    for name, item in metrics["models"].items():
        rows.append(
            {
                "Model": name,
                "CV ROC-AUC": item["cv"]["roc_auc_mean"],
                **{f"Test {m}": item["test"][m] for m in ["precision", "recall", "f1", "roc_auc"]},
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.image(str(REPORT_DIR / "figures" / "evaluation.png"))
    a, b = st.columns(2)
    a.image(str(REPORT_DIR / "figures" / "confusion_matrix.png"))
    with b:
        st.markdown("**Global feature sensitivity**")
        importance = pd.read_csv(REPORT_DIR / "feature_importance.csv")
        st.bar_chart(importance.head(10).set_index("feature")["roc_auc_drop"])
        st.caption(
            "Mean test ROC-AUC drop over five shuffled repeats. Correlated features can share "
            "importance. Used for inspection only, after selection."
        )
    st.write(
        "Precision: how many predicted churners churned. "
        "Recall: how many actual churners were found."
    )
    st.write("F1 balances precision and recall. ROC-AUC measures ranking across thresholds.")
    st.caption(
        "Probability calibration is inspected with reliability curves "
        "and Brier scores in metrics.json."
    )
    st.json(
        {"Test metrics for selected model": bundle.metadata["selected_test_metrics"]},
        expanded=False,
    )
