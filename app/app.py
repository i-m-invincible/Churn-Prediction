import os
import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import contextlib
import numpy

def card():
    """Fallback for older Streamlit without st.container(border=True)."""
    try:
        return st.container(border=True)
    except TypeError:
        return contextlib.nullcontext()


# ------------------------------------------------------------------ paths --
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "churn_model.pkl")
SCALER_PATH = os.path.join(BASE, "models", "scaler.pkl")
COLUMNS_PATH = os.path.join(BASE, "models", "model_columns.pkl")

NUM_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------- styling --
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp { background: #f6f7f9; }

header, #MainMenu, footer, [data-testid="stHeader"] { display: none; }

[data-testid="stMainBlockContainer"], .block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
}

/* ------------------------------ motion ------------------------------ */
@keyframes rise   { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
@keyframes grow   { from { width: 0; } }
@keyframes pulse  { 0%, 100% { opacity: 1; } 50% { opacity: .3; } }

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
}

/* ------------------------------ cards ------------------------------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid #e6e8ec !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 6px 16px rgba(15,23,42,.03) !important;
    animation: rise .55s cubic-bezier(.22,.61,.36,1) both;
}
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlock"] {
    gap: 0.75rem !important;
}

/* ------------------------------ widgets ----------------------------- */
[data-testid="stWidgetLabel"] p {
    font-size: .8rem !important;
    color: #475569 !important;
    font-weight: 500 !important;
}

div[data-baseweb="select"] > div,
.stNumberInput input {
    border-radius: 8px !important;
    transition: border-color .15s ease, box-shadow .15s ease !important;
}
div[data-baseweb="select"] > div:hover { border-color: #94a3b8 !important; }
.stNumberInput input:hover { border-color: #94a3b8 !important; }

/* ------------------------------ buttons ----------------------------- */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    box-shadow: none !important;
    transition: all .18s ease !important;
}
.stButton > button:hover  { transform: translateY(-1px); box-shadow: 0 3px 8px rgba(15,23,42,.10) !important; }
.stButton > button:active { transform: translateY(0) scale(.98); box-shadow: none !important; }

.stButton > button[kind="secondary"] {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    color: #374151 !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #f9fafb !important;
    border-color: #cbd5e1 !important;
    color: #111827 !important;
}

button[kind="primary"] {
    background: #111827 !important;
    border: 1px solid #111827 !important;
    color: #ffffff !important;
}
button[kind="primary"]:hover {
    background: #1e293b !important;
    border-color: #1e293b !important;
}
button[kind="primary"]:focus { box-shadow: none !important; }

/* ------------------------------ top bar ----------------------------- */
.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #ffffff;
    border-bottom: 1px solid #e6e8ec;
    padding: 0.85rem 1.5rem;
    margin: -1rem -1rem 1.5rem -1rem;
    animation: rise .45s ease both;
}
.brand {
    display: flex; align-items: center; gap: 0.6rem;
    font-weight: 600; font-size: 0.95rem; color: #111827;
}
.brand .logo {
    width: 26px; height: 26px; border-radius: 6px;
    background: #111827; display: flex; align-items: center;
    justify-content: center; color: #34d399; font-size: 13px;
}
.badge {
    background: #f3f4f6; color: #6b7280; border-radius: 4px;
    padding: 0.15rem 0.45rem; font-size: 0.7rem; font-weight: 500; margin-left: 0.4rem;
}
.status { color: #6b7280; font-size: 0.8rem; }
.status .dot { color: #10b981; animation: pulse 2.4s ease-in-out infinite; }

/* ------------------------------ header ------------------------------ */
.page-title {
    font-size: 1.5rem; font-weight: 700; color: #111827;
    margin: 0 0 0.25rem 0;
    animation: rise .5s ease both;
}
.page-sub { color: #6b7280; font-size: 0.9rem; animation: rise .5s .08s ease both; }

/* ------------------------------ text bits --------------------------- */
.section-label {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.05em;
    text-transform: uppercase; color: #9ca3af; margin-bottom: 0.85rem;
}

/* ------------------------------ risk -------------------------------- */
.risk-value {
    font-size: 2.9rem; font-weight: 700; line-height: 1;
    letter-spacing: -0.02em; font-variant-numeric: tabular-nums;
}
.risk-high { color: #dc2626; }
.risk-mod  { color: #d97706; }
.risk-low  { color: #059669; }

.pill {
    border-radius: 999px; padding: 0.22rem 0.7rem;
    font-size: 0.75rem; font-weight: 600;
    animation: rise .5s .15s ease both;
}
.pill-high { background: #fef2f2; color: #dc2626; }
.pill-mod  { background: #fffbeb; color: #d97706; }
.pill-low  { background: #ecfdf5; color: #059669; }

.progress-track {
    width: 100%; height: 6px; border-radius: 3px;
    background: #e5e7eb; margin: 0.85rem 0 0.4rem 0; overflow: hidden;
}
.progress-fill {
    height: 100%; border-radius: 3px;
    animation: grow .9s cubic-bezier(.22,.61,.36,1) both;
}

.scale-labels {
    display: flex; justify-content: space-between;
    color: #9ca3af; font-size: 0.72rem;
    font-variant-numeric: tabular-nums;
}

/* ------------------------------ metrics ----------------------------- */
.metric-row { display: flex; gap: 0.75rem; margin-top: 1.1rem; }
.metric {
    flex: 1; background: #f9fafb; border: 1px solid #f3f4f6;
    border-radius: 8px; padding: 0.7rem 0.9rem;
    transition: transform .18s ease, box-shadow .18s ease;
}
.metric:hover { transform: translateY(-2px); box-shadow: 0 3px 10px rgba(15,23,42,.06); }
.metric .label { color: #9ca3af; font-size: 0.72rem; font-weight: 500; margin-bottom: 0.2rem; }
.metric .value { font-weight: 600; font-size: 0.95rem; color: #111827; }

/* ------------------------------ snapshot ---------------------------- */
.snap-item {
    display: flex; justify-content: space-between; padding: 0.42rem 0;
    border-bottom: 1px solid #f3f4f6; font-size: 0.875rem;
    transition: padding-left .15s ease;
}
.snap-item:hover { padding-left: 4px; }
.snap-item:last-child { border-bottom: none; }
.snap-item .key { color: #6b7280; }
.snap-item .val { color: #111827; font-weight: 500; }

/* ------------------------------ factors ----------------------------- */
.factor {
    background: #f9fafb; border: 1px solid transparent; border-radius: 8px;
    padding: 0.7rem 0.9rem; margin-bottom: 0.5rem;
    display: flex; justify-content: space-between; align-items: flex-start; gap: 0.75rem;
    transition: background .15s ease, border-color .15s ease;
    animation: rise .45s ease both;
}
.factor:hover { background: #f3f4f6; border-color: #e5e7eb; }
.factor .title { font-weight: 600; font-size: 0.85rem; color: #111827; }
.factor .desc { color: #6b7280; font-size: 0.78rem; margin-top: 0.15rem; line-height: 1.35; }

.tag-up   { color: #dc2626; font-weight: 600; font-size: 0.75rem; white-space: nowrap; }
.tag-down { color: #059669; font-weight: 600; font-size: 0.75rem; white-space: nowrap; }

.model-note {
    color: #9ca3af; font-size: 0.75rem; display: flex; justify-content: space-between;
    padding-top: 0.75rem; margin-top: 0.5rem; border-top: 1px solid #f3f4f6;
}

.empty-state { color: #9ca3af; font-size: 0.9rem; line-height: 1.5; padding: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- top bar --
st.markdown("""
<div class="topbar">
  <div class="brand">
    <div class="logo">◆</div>
    Churn Intelligence
    <span class="badge">v1.0</span>
  </div>
  <div class="status">Model ready <span class="dot">●</span></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">Customer churn prediction</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ model --
missing = [p for p in (MODEL_PATH, SCALER_PATH, COLUMNS_PATH) if not os.path.exists(p)]
if missing:
    st.error("Missing model files: " + ", ".join(os.path.basename(m) for m in missing))
    st.info("Run `python train_model.py` first — it creates the models/ folder.")
    st.stop()


@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH), joblib.load(COLUMNS_PATH)


model, scaler, model_columns = load_artifacts()

FIELDS = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 12, "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "MonthlyCharges": 70.0, "TotalCharges": 840.0,
}

PROFILES = {
    "High risk": {
        **FIELDS, "tenure": 2, "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "TechSupport": "No",
        "MonthlyCharges": 95.0, "TotalCharges": 190.0, "PaperlessBilling": "Yes"
    },
    "Low risk": {
        **FIELDS, "tenure": 60, "Contract": "Two year",
        "PaymentMethod": "Credit card (automatic)", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "TechSupport": "Yes",
        "MonthlyCharges": 45.0, "TotalCharges": 2700.0, "PaperlessBilling": "No"
    },
    "Moderate risk": {
        **FIELDS, "tenure": 18, "Contract": "One year",
        "PaymentMethod": "Mailed check", "InternetService": "DSL",
        "OnlineSecurity": "No", "TechSupport": "Yes",
        "MonthlyCharges": 60.0, "TotalCharges": 1080.0, "PaperlessBilling": "Yes"
    },
}

for k, v in FIELDS.items():
    st.session_state.setdefault(k, v)


def apply_profile(name):
    for k, v in PROFILES[name].items():
        st.session_state[k] = v
    st.session_state["_active"] = name
    st.session_state.pop("result", None)


def reset_values():
    for k, v in FIELDS.items():
        st.session_state[k] = v
    st.session_state["_active"] = None
    st.session_state.pop("result", None)


FEATURE_INFO = {
    "Contract_Month-to-month": ("Contract term", "Month-to-month terms make it easy to leave.", None),
    "Contract_One year": ("Contract term", None, "Locked-in yearly terms reduce churn likelihood."),
    "Contract_Two year": ("Contract term", None, "Long-term commitment strongly reduces churn likelihood."),
    "InternetService_Fiber optic": ("Fiber optic internet", "Fiber accounts reflect higher price sensitivity and competitor switching.", None),
    "InternetService_DSL": ("DSL internet", None, "DSL customers show lower churn than fiber accounts."),
    "InternetService_No": ("No internet service", None, "No internet service is linked to lower churn."),
    "PaymentMethod_Electronic check": ("Payment method", "Manual billing is linked to higher voluntary churn.", None),
    "PaymentMethod_Mailed check": ("Payment method", "Manual billing is linked to higher voluntary churn.", None),
    "PaymentMethod_Bank transfer (automatic)": ("Payment method", None, "Automated billing reduces invoice friction and involuntary churn."),
    "PaymentMethod_Credit card (automatic)": ("Payment method", None, "Automated billing reduces invoice friction and involuntary churn."),
    "PaperlessBilling_Yes": ("Paperless billing", "Digital-only billing skews toward a more switch-prone segment.", None),
    "PaperlessBilling_No": ("Paperless billing", None, "Common among loyal long-tenure accounts."),
    "OnlineSecurity_Yes": ("Online security add-on", None, "Value-add security service increases platform stickiness."),
    "OnlineSecurity_No": ("No online security", "No security add-on reduces stickiness.", None),
    "OnlineBackup_Yes": ("Online backup add-on", None, "Value-add backup service increases platform stickiness."),
    "OnlineBackup_No": ("No online backup", "No backup add-on reduces stickiness.", None),
    "DeviceProtection_Yes": ("Device protection add-on", None, "Value-add protection increases platform stickiness."),
    "DeviceProtection_No": ("No device protection", "No protection add-on reduces stickiness.", None),
    "TechSupport_Yes": ("Tech support add-on", None, "Active tech support increases platform stickiness."),
    "TechSupport_No": ("No tech support", "No tech support reduces stickiness.", None),
    "StreamingTV_Yes": ("Streaming TV add-on", None, "Entertainment add-ons increase engagement and stickiness."),
    "StreamingMovies_Yes": ("Streaming movies add-on", None, "Entertainment add-ons increase engagement and stickiness."),
    "PhoneService_Yes": ("Phone service", None, "Basic phone service is broadly neutral for churn."),
    "PhoneService_No": ("No phone service", None, "Usually reflects a minimal service footprint."),
    "MultipleLines_Yes": ("Multiple lines", None, "Multi-line households show slightly higher stability."),
    "MultipleLines_No": ("Single line", None, "Single-line accounts are broadly neutral for churn."),
    "Partner_Yes": ("Has a partner", None, "Partnered households show more account stability."),
    "Partner_No": ("No partner", "Single-household accounts churn slightly more.", None),
    "Dependents_Yes": ("Has dependents", None, "Household dependents are linked to lower churn."),
    "Dependents_No": ("No dependents", "No dependents is linked to slightly higher churn.", None),
    "gender_Male": ("Gender", None, "Gender is not a meaningful churn driver."),
    "gender_Female": ("Gender", None, "Gender is not a meaningful churn driver."),
    "tenure": ("Account tenure", "Short tenure signals an early-lifecycle, higher-risk account.", "Longer tenure is one of the strongest loyalty signals."),
    "MonthlyCharges": ("Monthly bill size", "Higher monthly charges raise price-driven churn risk.", "Lower monthly charges reduce price-driven churn risk."),
    "TotalCharges": ("Total charges", "High lifetime billing on a young account signals price risk.", "Higher lifetime spend reflects an established relationship."),
    "SeniorCitizen": ("Senior citizen status", "Senior citizens show a modestly higher churn rate.", "Non-senior status is linked to lower churn."),
}


def describe(col, contribution):
    up = contribution > 0
    label, pos, neg = FEATURE_INFO.get(col, (None, None, None))
    if label is None:
        label = col.replace("_", " ")
        pos = f"{col} raises estimated churn risk."
        neg = f"{col} lowers estimated churn risk."
    return label, (pos or neg), up


# ---------------------------------------- subtitle + sample profile buttons --
head, b1, b2, b3, b4 = st.columns([2.7, 0.95, 0.95, 1.3, 0.9], gap="small")
with head:
    st.markdown(
        '<div class="page-sub">Estimate churn probability from account and service details.</div>',
        unsafe_allow_html=True,
    )

for col, name in zip((b1, b2, b3), PROFILES):
    with col:
        active = st.session_state.get("_active") == name
        if st.button(name, key=f"btn_{name}",
                     type="primary" if active else "secondary",
                     use_container_width=True):
            apply_profile(name)
            st.rerun()

with b4:
    if st.button("↻ Reset", key="btn_reset", use_container_width=True):
        reset_values()
        st.rerun()

# ---------------------------------------------------------------- layout --
left, right = st.columns([1.35, 1], gap="large")

with left:
    with card():
        st.markdown('<div class="section-label">Customer</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.selectbox("Gender", ["Male", "Female"], key="gender")
        with c2:
            st.selectbox("Senior citizen", [0, 1],
                         format_func=lambda v: "Yes" if v == 1 else "No",
                         key="SeniorCitizen")
        with c3:
            st.selectbox("Partner", ["Yes", "No"], key="Partner")
        with c4:
            st.selectbox("Dependents", ["Yes", "No"], key="Dependents")

        st.markdown('<div class="section-label" style="margin-top:1rem;">Account</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.number_input("Tenure (months)", min_value=0, max_value=100, key="tenure")
        with c2:
            st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="Contract")
        with c3:
            st.selectbox("Paperless billing", ["Yes", "No"], key="PaperlessBilling")
        with c4:
            st.selectbox(
                "Payment method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                key="PaymentMethod",
            )

        st.markdown('<div class="section-label" style="margin-top:1rem;">Services</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.selectbox("Phone service", ["Yes", "No"], key="PhoneService")
            st.selectbox("Online security", ["Yes", "No", "No internet service"], key="OnlineSecurity")
            st.selectbox("Tech support", ["Yes", "No", "No internet service"], key="TechSupport")
        with c2:
            st.selectbox("Multiple lines", ["Yes", "No", "No phone service"], key="MultipleLines")
            st.selectbox("Online backup", ["Yes", "No", "No internet service"], key="OnlineBackup")
            st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key="StreamingTV")
        with c3:
            st.selectbox("Internet service", ["DSL", "Fiber optic", "No"], key="InternetService")
            st.selectbox("Device protection", ["Yes", "No", "No internet service"], key="DeviceProtection")
            st.selectbox("Streaming movies", ["Yes", "No", "No internet service"], key="StreamingMovies")

        st.markdown('<div class="section-label" style="margin-top:1rem;">Billing</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Monthly charges ($)", min_value=0.0, key="MonthlyCharges")
        with c2:
            st.number_input("Total charges ($)", min_value=0.0, key="TotalCharges")

        predict = st.button("Predict churn", type="primary", use_container_width=True)

# ---------------------------------------------------------------- predict --
if predict:
    raw = pd.DataFrame([{k: st.session_state[k] for k in FIELDS}])
    cat_cols = raw.select_dtypes(include="object").columns.tolist()
    enc = pd.get_dummies(raw, columns=cat_cols, drop_first=False)
    enc = enc.reindex(columns=model_columns, fill_value=0).astype(float)
    enc[NUM_COLS] = scaler.transform(enc[NUM_COLS])

    pred = model.predict(enc)[0]
    prob = model.predict_proba(enc)[0][1]
    coefs = model.coef_[0]
    contribs = [
        (c, float(coefs[i] * enc.iloc[0][c]))
        for i, c in enumerate(model_columns)
        if abs(enc.iloc[0][c]) > 1e-6
    ]
    contribs.sort(key=lambda x: abs(x[1]), reverse=True)

    st.session_state["result"] = {
        "pred": int(pred),
        "prob": float(prob),
        "raw": raw.iloc[0].to_dict(),
        "factors": contribs[:4],
    }

# ---------------------------------------------------------------- results --
with right:
    with card():
        res = st.session_state.get("result")

        if not res:
            st.markdown('<div class="section-label">Churn risk</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="empty-state">Fill in the form and click <b>Predict churn</b> '
                'to see the risk assessment.</div>',
                unsafe_allow_html=True,
            )
        else:
            pct = res["prob"] * 100
            level = "high" if pct >= 50 else ("mod" if pct >= 25 else "low")
            color = {"high": "#dc2626", "mod": "#d97706", "low": "#059669"}[level]
            pill_cls, pill_txt = {
                "high": ("pill-high", "High probability"),
                "mod":  ("pill-mod",  "Moderate probability"),
                "low":  ("pill-low",  "Low probability"),
            }[level]

            st.markdown('<div class="section-label">Churn risk</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div class="risk-value risk-{level}" data-count="{pct:.1f}">{pct:.1f}%</div>
                <div class="pill {pill_cls}">{pill_txt}</div>
            </div>
            <div class="progress-track">
                <div class="progress-fill" style="width:{pct:.1f}%;background:{color};"></div>
            </div>
            <div class="scale-labels"><span>0%</span><span>50%</span><span>100%</span></div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric">
                    <div class="label">Prediction</div>
                    <div class="value">{"Likely to churn" if res["pred"] else "Likely to stay"}</div>
                </div>
                <div class="metric">
                    <div class="label">Confidence</div>
                    <div class="value"><span data-count="{pct:.1f}">{pct:.1f}%</span> probability</div>
                </div>
            </div>
            <p style="color:#9ca3af;font-size:0.8rem;margin:1rem 0 0 0;line-height:1.45;">
                Probabilistic estimate for decision support, not a guarantee.
            </p>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-label" style="margin-top:1.4rem;">Customer snapshot</div>', unsafe_allow_html=True)
            snap = res["raw"]
            for label, key in [
                ("Tenure", "tenure"),
                ("Contract", "Contract"),
                ("Monthly charges", "MonthlyCharges"),
                ("Internet service", "InternetService"),
                ("Payment method", "PaymentMethod"),
            ]:
                val = snap[key]
                if key == "MonthlyCharges":
                    val = f"${val:.2f}"
                elif key == "tenure":
                    val = f"{val} months"
                st.markdown(
                    f'<div class="snap-item"><span class="key">{label}</span><span class="val">{val}</span></div>',
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="section-label" style="margin-top:1.4rem;">Primary risk factors</div>', unsafe_allow_html=True)
            for i, (col, contrib) in enumerate(res["factors"]):
                label, desc, up = describe(col, contrib)
                tag = '<span class="tag-up">+ Risk</span>' if up else '<span class="tag-down">− Risk</span>'
                st.markdown(f"""
                <div class="factor" style="animation-delay:{0.08 * i:.2f}s;">
                    <div>
                        <div class="title">{label}</div>
                        <div class="desc">{desc}</div>
                    </div>
                    {tag}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div class="model-note">
                <span>Logistic regression</span>
                <span>Telco Customer Churn</span>
            </div>
            """, unsafe_allow_html=True)

            # ---- count-up animation (runs once per prediction) ----
            components.html("""
            <script>
            (function () {
                var tries = 0;
                (function run() {
                    var doc = window.parent.document;
                    var els = doc.querySelectorAll('[data-count]');
                    if (!els.length) {
                        if (++tries < 40) setTimeout(run, 60);
                        return;
                    }
                    els.forEach(function (el) {
                        var target = parseFloat(el.getAttribute('data-count'));
                        if (isNaN(target)) return;
                        var cur = parseFloat((el.textContent || '').replace(/[^\\d.]/g, ''));
                        var from = isNaN(cur) ? 0 : cur;
                        var dur = 850, t0 = null;
                        function step(ts) {
                            if (!t0) t0 = ts;
                            var p = Math.min((ts - t0) / dur, 1);
                            var e = 1 - Math.pow(1 - p, 3);        // ease-out cubic
                            el.textContent = (from + (target - from) * e).toFixed(1) + '%';
                            if (p < 1) requestAnimationFrame(step);
                        }
                        requestAnimationFrame(step);
                    });
                })();
            })();
            </script>
            """, height=0)