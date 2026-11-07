"""Native Streamlit views; inference only, with no training at application startup."""

import json
import logging

import altair as alt
import pandas as pd
import streamlit as st

from telco_churn.batch import prediction_csv, read_customer_csv
from telco_churn.data import load_dataset
from telco_churn.inference import explain_customer, load_bundle, predict_customers
from telco_churn.schema import ARTIFACT_DIR, CATEGORIES, INTERNET_ADDONS, REPORT_DIR, ROOT

LOGGER = logging.getLogger(__name__)
TEAL = "#167D8D"
AMBER = "#C68B40"
WORKSPACES = ["Predict Churn", "Explore Data", "Model Performance"]
DEFAULT_CUSTOMER = {
    "tenure": 3,
    "MonthlyCharges": 95.0,
    "TotalCharges": 285.0,
    "gender": "Female",
    "SeniorCitizen": "0",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    **{name: "No" for name in INTERNET_ADDONS},
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}
EXAMPLES = {
    "New subscriber": DEFAULT_CUSTOMER,
    "Established customer": {
        **DEFAULT_CUSTOMER,
        "tenure": 60,
        "MonthlyCharges": 65.0,
        "TotalCharges": 3900.0,
        "Contract": "Two year",
        "InternetService": "DSL",
        "Partner": "Yes",
        "OnlineSecurity": "Yes",
        "TechSupport": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
    },
}


@st.cache_resource(max_entries=1, show_spinner="Loading the verified model…")
def resources(metadata_revision: str):
    return load_bundle()


@st.cache_data(max_entries=1, show_spinner=False)
def report_data(metadata_revision: str):
    metrics = json.loads((REPORT_DIR / "metrics.json").read_text(encoding="utf-8"))
    features, target, ids = load_dataset()
    manifest = pd.read_csv(REPORT_DIR / "split_manifest.csv")
    train_ids = set(manifest.loc[manifest["partition"].eq("train"), "customerID"])
    mask = ids.isin(train_ids)
    development = features.loc[mask].assign(Churn=target.loc[mask])
    importance = pd.read_csv(REPORT_DIR / "feature_importance.csv")
    return metrics, development, importance


def apply_example(name: str):
    for field, value in EXAMPLES[name].items():
        st.session_state[f"customer_{field}"] = value
    st.session_state["total_unknown"] = False
    st.session_state["assessment"] = None
    st.session_state["assessment_label"] = f"Example profile · {name}"


def score_profile(bundle, raw: dict):
    frame = pd.DataFrame([raw])
    result = predict_customers(bundle, frame).iloc[0].to_dict()
    factors = explain_customer(bundle, frame)
    st.session_state["assessment"] = (result, factors, raw.copy())


def customer_form(bundle):
    for field, value in DEFAULT_CUSTOMER.items():
        st.session_state.setdefault(f"customer_{field}", value)
    st.session_state.setdefault("total_unknown", False)
    st.session_state.setdefault("assessment_label", "Example profile · New subscriber")
    st.subheader("Customer profile")
    st.caption("Try an example or enter a customer's account details.")
    example_columns = st.columns(2)
    for column, name in zip(example_columns, EXAMPLES, strict=True):
        column.button(name, on_click=apply_example, args=(name,), width="stretch")
    a, b = st.columns(2)
    phone = a.selectbox("Phone service", ["Yes", "No"], key="customer_PhoneService")
    internet = b.selectbox(
        "Internet service", ["Fiber optic", "DSL", "No"], key="customer_InternetService"
    )
    lines = st.session_state["customer_MultipleLines"]
    if phone == "No":
        st.session_state["customer_MultipleLines"] = "No phone service"
    elif lines == "No phone service":
        st.session_state["customer_MultipleLines"] = "No"
    for name in INTERNET_ADDONS:
        key = f"customer_{name}"
        if internet == "No":
            st.session_state[key] = "No internet service"
        elif st.session_state[key] == "No internet service":
            st.session_state[key] = "No"
    with st.form("customer_form", border=False):
        raw = {"PhoneService": phone, "InternetService": internet}
        account, services, customer = st.tabs(["Account", "Services", "Customer"])
        with account:
            a, b = st.columns(2)
            raw["tenure"] = a.number_input("Tenure (months)", 0, 120, key="customer_tenure")
            raw["Contract"] = b.selectbox(
                "Contract", CATEGORIES["Contract"], key="customer_Contract"
            )
            raw["MonthlyCharges"] = a.number_input(
                "Monthly charges ($)", 0.0, 1000.0, key="customer_MonthlyCharges"
            )
            raw["TotalCharges"] = b.number_input(
                "Total charges ($)", 0.0, 120000.0, key="customer_TotalCharges"
            )
            unknown = st.checkbox(
                "Total charges are unavailable",
                key="total_unknown",
                help="Missing charges use the median learned from training data.",
            )
            raw["PaymentMethod"] = st.selectbox(
                "Payment method", CATEGORIES["PaymentMethod"], key="customer_PaymentMethod"
            )
            raw["PaperlessBilling"] = st.selectbox(
                "Paperless billing", ["Yes", "No"], key="customer_PaperlessBilling"
            )
        with services:
            raw["MultipleLines"] = st.selectbox(
                "Multiple lines",
                ["No", "Yes"] if phone == "Yes" else ["No phone service"],
                key="customer_MultipleLines",
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
                    key=f"customer_{name}",
                )
        with customer:
            a, b = st.columns(2)
            raw["gender"] = a.selectbox("Gender", CATEGORIES["gender"], key="customer_gender")
            raw["SeniorCitizen"] = b.selectbox(
                "Senior citizen",
                CATEGORIES["SeniorCitizen"],
                format_func=lambda x: "Yes" if x == "1" else "No",
                key="customer_SeniorCitizen",
            )
            raw["Partner"] = a.selectbox("Partner", ["No", "Yes"], key="customer_Partner")
            raw["Dependents"] = b.selectbox("Dependents", ["No", "Yes"], key="customer_Dependents")
        submitted = st.form_submit_button("Predict churn", type="primary", width="stretch")
    if submitted:
        if unknown:
            raw["TotalCharges"] = None
        try:
            score_profile(bundle, raw)
            st.session_state["assessment_label"] = "Your submitted profile"
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            LOGGER.exception("Prediction failed")
            st.error("This profile could not be scored. Please try again.")
    elif not st.session_state.get("assessment"):
        if unknown:
            raw["TotalCharges"] = None
        score_profile(bundle, raw)


def risk_assessment(bundle):
    st.subheader("Risk assessment")
    st.caption(st.session_state["assessment_label"])
    assessment = st.session_state.get("assessment")
    if not assessment:
        st.info("Submit a valid customer profile to see an assessment.")
        return
    result, factors, snapshot = assessment
    p = float(result["churn_probability"])
    st.metric("Estimated churn probability", f"{p:.1%}")
    st.progress(p)
    a, b = st.columns(2)
    a.metric("Risk category", result["risk_band"], border=True)
    b.metric("Predicted class", "Churn" if result["predicted_churn"] else "Stay", border=True)
    st.caption("Risk: low <30% · moderate 30–60% · high ≥60%. Class threshold: 50%.")
    st.markdown("**What affects this estimate**")
    meaningful = factors.loc[factors["probability_difference_pp"].abs().gt(0.01)].head(5)
    if meaningful.empty:
        st.info("This profile matches the reference on the compared factors.")
    else:
        chart = (
            alt.Chart(meaningful)
            .mark_bar(cornerRadiusEnd=3, size=15)
            .encode(
                x=alt.X(
                    "probability_difference_pp:Q",
                    title="Probability difference (percentage points)",
                ),
                y=alt.Y(
                    "factor:N",
                    sort=meaningful["factor"].tolist(),
                    title=None,
                    axis=alt.Axis(labelLimit=160),
                ),
                color=alt.condition(
                    alt.datum.probability_difference_pp > 0, alt.value(AMBER), alt.value(TEAL)
                ),
                tooltip=[
                    alt.Tooltip("factor:N", title="Factor"),
                    alt.Tooltip(
                        "probability_difference_pp:Q", title="Difference (pp)", format="+.1f"
                    ),
                ],
            )
            .properties(height=180)
        )
        zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(color="#A8B4BC").encode(x="zero:Q")
        st.altair_chart((chart + zero).configure_view(stroke=None), width="stretch")
    st.caption(
        "Positive bars mean higher estimated risk than the reference value. Each factor group "
        "is replaced separately; differences do not add up and do not establish causes."
    )
    with st.expander("Inspect the comparison and submitted profile"):
        st.dataframe(
            factors.rename(
                columns={
                    "factor": "Factor",
                    "customer_value": "Customer",
                    "reference_value": "Reference",
                    "probability_difference_pp": "Difference (pp)",
                    "replacement_probability": "Probability after replacement",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.json(snapshot, expanded=False)
        st.caption(f"Inference uses the saved {bundle.metadata['selected_model']} pipeline.")
    st.caption("Results refer to the last submitted profile. Submit again after editing.")


def batch_predictions(bundle):
    with st.expander("Batch prediction · upload a customer CSV"):
        st.write("Score up to 10,000 customers using the same verified model and preprocessing.")
        st.download_button(
            "Download example CSV",
            (ROOT / "examples" / "customer_batch.csv").read_bytes(),
            "customer_batch.csv",
            "text/csv",
            on_click="ignore",
        )
        uploaded = st.file_uploader(
            "Customer CSV", type=["csv"], help="UTF-8 CSV · maximum 5 MB · 19 raw predictor columns"
        )
        st.caption("Use sample or anonymized data. Uploads are not written to the repository.")
        if uploaded is None:
            return
        try:
            frame = read_customer_csv(uploaded.getvalue())
            output = predict_customers(bundle, frame)
            if "customerID" in frame:
                output.insert(0, "customerID", frame["customerID"])
            a, b, c = st.columns(3)
            a.metric("Customers scored", f"{len(output):,}")
            b.metric("High risk", f"{output['risk_band'].eq('High').sum():,}")
            c.metric("Mean probability", f"{output['churn_probability'].mean():.1%}")
            st.dataframe(
                output,
                hide_index=True,
                width="stretch",
                column_config={
                    "churn_probability": st.column_config.ProgressColumn(
                        "Churn probability", min_value=0, max_value=1, format="percent"
                    )
                },
            )
            st.download_button(
                "Download predictions",
                prediction_csv(output),
                "churn_predictions.csv",
                "text/csv",
                on_click="ignore",
            )
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            LOGGER.exception("Batch prediction failed")
            st.error("The CSV could not be scored. Please check the example format and try again.")


def predict_workspace(bundle):
    st.header("Predict customer churn")
    st.write("Turn a customer profile into a clear, explainable risk estimate.")
    left, right = st.columns([1.15, 1], gap="large")
    with left, st.container(border=True):
        customer_form(bundle)
    with right, st.container(border=True):
        risk_assessment(bundle)
    batch_predictions(bundle)
    st.caption(
        "Educational estimates from a historical sample dataset. Risk bands are illustrative."
    )


def group_chart(frame: pd.DataFrame, feature: str):
    rates = frame.groupby(feature, observed=True)["Churn"].agg(["mean", "count"]).reset_index()
    return alt.Chart(rates).mark_bar(color=TEAL, cornerRadiusEnd=4).encode(
        x=alt.X("mean:Q", title="Observed churn rate", axis=alt.Axis(format=".0%")),
        y=alt.Y(f"{feature}:N", title=None, sort="-x", axis=alt.Axis(labelLimit=240)),
        tooltip=[
            alt.Tooltip(f"{feature}:N", title="Group"),
            alt.Tooltip("mean:Q", title="Churn rate", format=".1%"),
            alt.Tooltip("count:Q", title="Customers", format=","),
        ],
    ).properties(height=250).configure_view(stroke=None), rates


def explore_workspace(frame):
    st.header("Explore the customer data")
    st.write("Understand the patterns behind the model using the training partition only.")
    a, b, c = st.columns(3)
    a.metric("Training customers", f"{len(frame):,}", border=True)
    b.metric("Observed churn rate", f"{frame['Churn'].mean():.1%}", border=True)
    c.metric("Missing total charges", f"{frame['TotalCharges'].isna().sum():,}", border=True)
    with st.container(border=True):
        st.subheader("Compare customer groups")
        feature = st.selectbox(
            "Group by",
            ["Contract", "InternetService", "PaymentMethod", "SeniorCitizen", "TechSupport"],
        )
        chart, rates = group_chart(frame, feature)
        st.altair_chart(chart, width="stretch")
        with st.expander("View the underlying counts"):
            st.dataframe(
                rates.rename(columns={"mean": "Churn rate", "count": "Customers"}),
                hide_index=True,
                width="stretch",
            )
    with st.expander("Tenure, target balance and numeric correlations"):
        st.image(str(REPORT_DIR / "figures" / "eda_overview.png"))
        st.image(str(REPORT_DIR / "figures" / "correlations.png"))
    st.caption("These are descriptive associations. The held-out test set is excluded from EDA.")


def performance_workspace(bundle, metrics, importance):
    st.header("Model performance")
    st.write("Three classifiers, one fixed test partition, a transparent selection rule.")
    selected = metrics["models"][metrics["selected_model"]]
    a, b, c = st.columns(3)
    a.metric("Selected model", metrics["selected_model"], border=True)
    b.metric("Test ROC-AUC", f"{selected['test']['roc_auc']:.4f}", border=True)
    c.metric("Held-out customers", f"{metrics['test_customers']:,}", border=True)
    st.caption(
        f"Selected by mean {metrics['cv_folds']}-fold training-only CV ROC-AUC. "
        "Test scores are calculated afterward; the CV differences are small."
    )
    rows = [
        {
            "Model": name,
            "CV ROC-AUC": item["cv"]["roc_auc_mean"],
            **{f"Test {m}": item["test"][m] for m in ["precision", "recall", "f1", "roc_auc"]},
        }
        for name, item in metrics["models"].items()
    ]
    st.dataframe(
        pd.DataFrame(rows).style.format(
            {column: "{:.4f}" for column in rows[0] if column != "Model"}
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Precision, recall and F1 refer to Churn = 1, with a fixed 50% classification threshold."
    )
    curves, factors = st.tabs(["ROC & calibration", "Confusion matrix & factors"])
    with curves:
        st.image(str(REPORT_DIR / "figures" / "evaluation.png"))
        st.caption(
            "ROC-AUC measures ranking. Reliability curves compare probabilities with outcomes."
        )
    with factors:
        a, b = st.columns(2)
        a.image(str(REPORT_DIR / "figures" / "confusion_matrix.png"))
        with b:
            st.markdown("**Global feature sensitivity**")
            chart = (
                alt.Chart(importance.head(10))
                .mark_bar(color=TEAL)
                .encode(
                    x=alt.X("roc_auc_drop:Q", title="Mean test ROC-AUC drop"),
                    y=alt.Y("feature:N", sort="-x", title=None),
                    tooltip=[
                        "feature:N",
                        alt.Tooltip("roc_auc_drop:Q", format=".4f"),
                        alt.Tooltip("std:Q", format=".4f"),
                    ],
                )
                .properties(height=280)
                .configure_view(stroke=None)
            )
            st.altair_chart(chart, width="stretch")
            st.caption(
                "Five shuffled repeats, inspected after selection. "
                "Correlated inputs can share importance."
            )
    with st.expander("Full metrics and evaluation notes"):
        st.json(bundle.metadata["selected_test_metrics"])
        st.write(
            "The served model is the same training-fitted model evaluated here. "
            "It has not been refitted on test data."
        )


def main():
    st.set_page_config(
        page_title="Churn Atlas | Customer Churn Prediction",
        page_icon=":material/query_stats:",
        layout="wide",
    )
    try:
        revision = (ARTIFACT_DIR / "metadata.json").read_text(encoding="utf-8")
        bundle = resources(revision)
        metrics, development, importance = report_data(revision)
    except Exception:
        LOGGER.exception("Verified application resources could not be loaded")
        st.title("Churn Atlas")
        st.error("The model is temporarily unavailable. Please try again later.")
        st.caption("The maintainer can inspect the application logs for deployment details.")
        return
    with st.sidebar:
        st.title("Churn Atlas")
        st.caption("CUSTOMER RETENTION INTELLIGENCE")
        page = st.radio("Workspace", WORKSPACES, label_visibility="collapsed")
        st.divider()
        st.markdown(f"**{bundle.metadata['selected_model']}** · verified saved model")
        st.caption(
            f"{metrics['train_customers']:,} training customers · "
            f"{metrics['test_customers']:,} held-out test customers"
        )
        st.markdown("[GitHub repository](https://github.com/i-m-invincible/Churn-Prediction)")
        st.caption("Built by Bishal Dutta")
    try:
        if page == "Predict Churn":
            predict_workspace(bundle)
        elif page == "Explore Data":
            explore_workspace(development)
        else:
            performance_workspace(bundle, metrics, importance)
    except Exception:
        LOGGER.exception("Dashboard view failed")
        st.error("This view is temporarily unavailable. Please try another workspace or reload.")
