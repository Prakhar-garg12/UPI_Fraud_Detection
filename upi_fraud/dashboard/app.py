# ============================================================
#  UPI Fraud Detection — Streamlit Dashboard
#  Research Paper Flow (IJEDR 2026):
#  7 Modules visualized: Data Collection → Preprocessing →
#  Feature Extraction → ML Training → Fraud Detection →
#  Alert & Notification → Reporting & Monitoring
#
#  Run: streamlit run dashboard/app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import random
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# ─── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="UPI Fraud Guard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .block-title {
        font-size: 12px; font-weight: 600; color: #6c757d;
        text-transform: uppercase; letter-spacing: 0.06em;
        margin-bottom: 0.4rem;
    }
    .module-badge {
        background: #e8f4fd; color: #1565c0; padding: 3px 10px;
        border-radius: 12px; font-size: 11px; font-weight: 600;
        border: 1px solid #90caf9;
    }
    .allow-badge  { background:#d4edda;color:#155724;padding:6px 18px;
                    border-radius:20px;font-weight:700;font-size:18px; }
    .review-badge { background:#fff3cd;color:#856404;padding:6px 18px;
                    border-radius:20px;font-weight:700;font-size:18px; }
    .block-badge  { background:#f8d7da;color:#721c24;padding:6px 18px;
                    border-radius:20px;font-weight:700;font-size:18px; }
    div[data-testid="stMetric"] {
        background:white;border-radius:10px;
        padding:12px 16px;border:1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load model artifacts ─────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mdir  = os.path.join(base, "models")

    best_model      = joblib.load(os.path.join(mdir, "best_model.pkl"))
    best_model_name = joblib.load(os.path.join(mdir, "best_model_name.pkl"))
    threshold       = joblib.load(os.path.join(mdir, "threshold.pkl"))
    encoders        = joblib.load(os.path.join(mdir, "label_encoders.pkl"))
    features        = joblib.load(os.path.join(mdir, "feature_names.pkl"))
    scaler          = joblib.load(os.path.join(mdir, "scaler.pkl"))

    # Load all 3 models for model comparison page
    lr_model = joblib.load(os.path.join(mdir, "logistic_regression_model.pkl"))
    rf_model = joblib.load(os.path.join(mdir, "random_forest_model.pkl"))
    gb_model = joblib.load(os.path.join(mdir, "gradient_boosting_model.pkl"))

    # Load performance report
    report_path = os.path.join(base, "reports", "model_performance_report.json")
    report = {}
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)

    return (best_model, best_model_name, threshold, encoders,
            features, scaler, lr_model, rf_model, gb_model, report)

(best_model, BEST_MODEL_NAME, THRESHOLD, encoders,
 FEATURES, scaler, lr_model, rf_model, gb_model, REPORT) = load_artifacts()

# ─── Constants ────────────────────────────────────────────────
CITIES     = ['Mumbai','Delhi','Bangalore','Hyderabad','Chennai','Kolkata',
              'Pune','Ahmedabad','Jaipur','Surat','Lucknow','Indore',
              'Bhopal','Nagpur','Patna','Coimbatore','Bhubaneswar']
TX_TYPES   = ['P2P','P2M','Bill','Recharge']
MERCH_CATS = ['food','grocery','travel','entertainment',
              'utility','retail','healthcare','peer']


# ─── Helpers ──────────────────────────────────────────────────
def encode_and_predict(row: dict, model, use_scaled=False):
    """Step 2+3+4: Preprocess → Extract features → Predict."""
    r = row.copy()
    for col in ["transaction_type","transaction_city","merchant_category"]:
        le  = encoders[col]
        val = r[col]
        r[col] = int(le.transform([val])[0]) if val in le.classes_ else 0
    vec = np.array([[r[f] for f in FEATURES]], dtype=float)
    if use_scaled:
        vec = scaler.transform(vec)
    return float(model.predict_proba(vec)[0][1])


def predict_all_models(row: dict):
    """Run all 3 paper models and return individual + ensemble probs."""
    lr_prob = encode_and_predict(row, lr_model, use_scaled=True)
    rf_prob = encode_and_predict(row, rf_model, use_scaled=False)
    gb_prob = encode_and_predict(row, gb_model, use_scaled=False)
    # Best model prediction
    use_sc  = "Logistic" in BEST_MODEL_NAME
    best_prob = encode_and_predict(row, best_model, use_scaled=use_sc)
    return lr_prob, rf_prob, gb_prob, best_prob


def get_decision(prob):
    if prob >= 0.60:   return "BLOCK",  "HIGH"
    elif prob >= THRESHOLD: return "REVIEW", "MEDIUM"
    else:              return "ALLOW",  "LOW"


def get_risk_factors(row: dict, prob: float):
    """Step 3: Feature Extraction — red flag behavioral indicators."""
    flags = []
    if row["ip_risk_score"]          > 0.5:  flags.append("High IP risk score")
    if row["sim_change_flag"]        == 1:   flags.append("SIM card recently changed")
    if row["pin_attempts"]           >= 3:   flags.append("Multiple PIN attempts")
    if row["is_new_device"]          == 1:   flags.append("Unrecognized device")
    if row["location_mismatch"]      == 1:   flags.append("Location mismatch")
    if row["txns_last_1hr"]          >= 5:   flags.append(f"High velocity ({row['txns_last_1hr']}/hr)")
    if row["kyc_status"]             == 2:   flags.append("KYC rejected")
    if row["failed_txns_24hr"]       >= 3:   flags.append("Multiple failed txns today")
    if abs(row["amount_deviation_from_avg"]) > 2: flags.append("Unusual spending amount")
    return flags[:3] if flags else (["Anomalous pattern detected"] if prob >= THRESHOLD else [])


def generate_alert_message(decision: str):
    """Step 6: Alert & Notification Module messages."""
    if decision == "BLOCK":
        return ("🚨 User Alert: Transaction BLOCKED — verify immediately",
                "🏦 Bank Alert: Block & require additional authentication")
    elif decision == "REVIEW":
        return ("⚠️ User Alert: Please verify this transaction in your UPI app",
                "🏦 Bank Alert: Flag for review — suspicious activity detected")
    return ("✅ Transaction cleared — no alerts sent", "")


def random_transaction(fraud=False):
    """Synthetic transaction for live feed simulation."""
    hour = random.randint(0, 23)
    if fraud:
        return {
            "transaction_type"          : random.choice(TX_TYPES),
            "transaction_city"          : random.choice(CITIES),
            "amount_inr"                : round(random.uniform(5000, 50000), 2),
            "hour_of_day"               : hour,
            "day_of_week"               : random.randint(0, 6),
            "is_odd_hour"               : int(hour < 6 or hour > 22),
            "is_new_device"             : random.choices([0,1],[0.2,0.8])[0],
            "sim_change_flag"           : random.choices([0,1],[0.2,0.8])[0],
            "kyc_status"                : random.choices([0,1,2],[0.1,0.2,0.7])[0],
            "location_mismatch"         : random.choices([0,1],[0.2,0.8])[0],
            "is_international"          : random.choices([0,1],[0.5,0.5])[0],
            "is_new_payee"              : random.choices([0,1],[0.3,0.7])[0],
            "txns_last_1hr"             : random.randint(5, 15),
            "txns_last_24hr"            : random.randint(10, 30),
            "pin_attempts"              : random.choices([1,2,3],[0.1,0.2,0.7])[0],
            "ip_risk_score"             : round(random.uniform(0.5, 0.99), 3),
            "failed_txns_24hr"          : random.randint(2, 8),
            "amount_deviation_from_avg" : round(random.uniform(2.0, 5.0), 3),
            "txn_velocity_score"        : round(random.uniform(0.6, 0.99), 3),
            "avg_txn_amount_30d"        : round(random.uniform(500, 2000), 2),
            "upi_handle_risk"           : round(random.uniform(0.5, 0.99), 3),
            "merchant_category"         : random.choice(MERCH_CATS),
        }
    return {
        "transaction_type"          : random.choice(TX_TYPES),
        "transaction_city"          : random.choice(CITIES),
        "amount_inr"                : round(random.uniform(100, 5000), 2),
        "hour_of_day"               : hour,
        "day_of_week"               : random.randint(0, 6),
        "is_odd_hour"               : int(hour < 6 or hour > 22),
        "is_new_device"             : random.choices([0,1],[0.95,0.05])[0],
        "sim_change_flag"           : 0,
        "kyc_status"                : 1,
        "location_mismatch"         : random.choices([0,1],[0.95,0.05])[0],
        "is_international"          : 0,
        "is_new_payee"              : random.choices([0,1],[0.8,0.2])[0],
        "txns_last_1hr"             : random.randint(0, 3),
        "txns_last_24hr"            : random.randint(1, 8),
        "pin_attempts"              : random.choices([1,2,3],[0.88,0.09,0.03])[0],
        "ip_risk_score"             : round(random.uniform(0.0, 0.2), 3),
        "failed_txns_24hr"          : random.choices([0,1],[0.92,0.08])[0],
        "amount_deviation_from_avg" : round(random.uniform(-0.5, 0.5), 3),
        "txn_velocity_score"        : round(random.uniform(0.0, 0.2), 3),
        "avg_txn_amount_30d"        : round(random.uniform(2000, 8000), 2),
        "upi_handle_risk"           : round(random.uniform(0.0, 0.1), 3),
        "merchant_category"         : random.choice(MERCH_CATS),
    }


# ─── Session state ────────────────────────────────────────────
for key, default in [("live_feed", []), ("live_running", False),
                     ("total_txns", 0), ("total_fraud", 0), ("total_blocked", 0)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/UPI-Logo-vector.svg/320px-UPI-Logo-vector.svg.png",
             width=110)
    st.title("🛡️ UPI Fraud Guard")
    st.caption("IJEDR 2026 — Research Paper Implementation")
    st.markdown("---")

    page = st.radio("Navigation", [
        "🔍 Check Transaction",
        "📊 Model Comparison",
        "⚡ Live Feed",
        "📈 Analytics",
        "🗺️ System Pipeline",
    ])
    st.markdown("---")
    st.caption(f"Best model  : **{BEST_MODEL_NAME}**")
    st.caption(f"Threshold   : **{THRESHOLD:.4f}**")
    st.caption(f"Session txns: **{st.session_state.total_txns}**")


# ══════════════════════════════════════════════════════════════
# PAGE 1 — Check Transaction (Full 7-Module Pipeline)
# ══════════════════════════════════════════════════════════════
if page == "🔍 Check Transaction":
    st.title("🔍 Check Transaction")
    st.markdown(
        '<span class="module-badge">Step 1–6: Data Collection → Preprocessing → '
        'Feature Extraction → ML Prediction → Risk Score → Alert</span>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ── Step 1: Data Collection — Transaction Parameters ──
    st.markdown("##### Step 1 — Transaction Parameters (Data Collection)")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<p class="block-title">Transaction Info</p>', unsafe_allow_html=True)
        tx_type   = st.selectbox("Transaction type",  TX_TYPES)
        tx_city   = st.selectbox("City",              CITIES)
        merch_cat = st.selectbox("Merchant category", MERCH_CATS)
        amount    = st.number_input("Amount (₹)", min_value=1.0, value=2000.0, step=100.0)
        hour      = st.slider("Hour of day", 0, 23, 14)
        dow       = st.slider("Day of week (0=Mon)", 0, 6, 2)

    with col2:
        st.markdown('<p class="block-title">Device & Location</p>', unsafe_allow_html=True)
        is_new_device  = st.toggle("New / unrecognized device", value=False)
        sim_change     = st.toggle("SIM card recently changed", value=False)
        loc_mismatch   = st.toggle("Location mismatch",         value=False)
        is_intl        = st.toggle("International transaction",  value=False)
        kyc_status     = st.selectbox("KYC status",
                            options=[0,1,2],
                            format_func=lambda x: {0:"Pending",1:"Verified",2:"Rejected"}[x],
                            index=1)

    with col3:
        st.markdown('<p class="block-title">Behavioral History</p>', unsafe_allow_html=True)
        is_new_payee    = st.toggle("New payee",                value=False)
        txns_1hr        = st.slider("Txns in last 1 hr",        0, 15, 1)
        txns_24hr       = st.slider("Txns in last 24 hr",       0, 50, 5)
        pin_attempts    = st.slider("PIN attempts",              1,  3, 1)
        ip_risk         = st.slider("IP risk score",           0.0,1.0,0.1,step=0.01)
        failed_txns     = st.slider("Failed txns today",        0,  10, 0)
        amt_deviation   = st.slider("Amount deviation (σ)", -5.0, 5.0, 0.0, step=0.1)
        velocity_score  = st.slider("Velocity score",         0.0, 1.0, 0.1, step=0.01)
        avg_30d         = st.number_input("Avg 30d spend (₹)", min_value=0.0, value=3000.0, step=100.0)
        upi_handle_risk = st.slider("UPI handle risk",        0.0, 1.0, 0.05, step=0.01)

    st.markdown("")
    predict_btn = st.button("🔎 Analyze Transaction", type="primary", use_container_width=True)

    if predict_btn:
        row = {
            "transaction_type"          : tx_type,
            "transaction_city"          : tx_city,
            "amount_inr"                : amount,
            "hour_of_day"               : hour,
            "day_of_week"               : dow,
            "is_odd_hour"               : int(hour < 6 or hour > 22),
            "is_new_device"             : int(is_new_device),
            "sim_change_flag"           : int(sim_change),
            "kyc_status"                : kyc_status,
            "location_mismatch"         : int(loc_mismatch),
            "is_international"          : int(is_intl),
            "is_new_payee"              : int(is_new_payee),
            "txns_last_1hr"             : txns_1hr,
            "txns_last_24hr"            : txns_24hr,
            "pin_attempts"              : pin_attempts,
            "ip_risk_score"             : ip_risk,
            "failed_txns_24hr"          : failed_txns,
            "amount_deviation_from_avg" : amt_deviation,
            "txn_velocity_score"        : velocity_score,
            "avg_txn_amount_30d"        : avg_30d,
            "upi_handle_risk"           : upi_handle_risk,
            "merchant_category"         : merch_cat,
        }

        with st.spinner("Running fraud detection pipeline..."):
            t0 = time.time()
            lr_prob, rf_prob, gb_prob, best_prob = predict_all_models(row)
            ms = round((time.time() - t0)*1000, 1)

        decision, risk_level = get_decision(best_prob)
        risk_factors = get_risk_factors(row, best_prob)
        user_alert, bank_alert = generate_alert_message(decision)

        st.markdown("---")
        # ── Step 5: Results ──
        st.markdown("##### Step 5 — Fraud Detection Module Results")
        r1, r2, r3, r4 = st.columns(4)
        badge = {"ALLOW":"allow-badge","REVIEW":"review-badge","BLOCK":"block-badge"}[decision]
        r1.markdown(f'<div style="text-align:center"><p class="block-title">Decision</p>'
                    f'<span class="{badge}">{decision}</span></div>', unsafe_allow_html=True)
        r2.metric("Fraud Probability", f"{best_prob*100:.1f}%")
        r3.metric("Risk Level", risk_level)
        r4.metric("Response Time", f"{ms} ms")

        # Risk gauge
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(best_prob*100, 1),
            title={"text": f"Fraud Risk Score ({BEST_MODEL_NAME})"},
            gauge={
                "axis"     : {"range": [0,100]},
                "bar"      : {"color": "#E24B4A" if best_prob > 0.5 else "#1D9E75"},
                "steps"    : [{"range":[0,30],"color":"#d4edda"},
                               {"range":[30,60],"color":"#fff3cd"},
                               {"range":[60,100],"color":"#f8d7da"}],
                "threshold": {"line":{"color":"black","width":3},
                               "value":round(THRESHOLD*100,1)},
            },
            number={"suffix":"%"},
        ))
        fig_g.update_layout(height=260, margin=dict(t=30,b=10,l=20,r=20))
        st.plotly_chart(fig_g, use_container_width=True)

        # ── Step 4 side-by-side model comparison ──
        st.markdown("##### Step 4 — ML Model Comparison (All 3 Models)")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Logistic Regression",  f"{lr_prob*100:.1f}%",
                   help="Best for basic statistical classification (paper)")
        mc2.metric("Random Forest",        f"{rf_prob*100:.1f}%",
                   help="Most stable — combines multiple behavioural features (paper)")
        mc3.metric("Gradient Boosting",    f"{gb_prob*100:.1f}%",
                   help="Detects delicate fraudulent signals (paper)")

        # ── Step 3: Red flag indicators ──
        if risk_factors:
            st.markdown("##### Step 3 — Feature Extraction: Red Flag Indicators")
            for f in risk_factors:
                icon = "🔴" if decision == "BLOCK" else "🟡"
                st.markdown(f"{icon} {f}")

        # ── Step 6: Alert & Notification ──
        st.markdown("##### Step 6 — Alert & Notification Module")
        if decision != "ALLOW":
            st.warning(user_alert)
            st.error(bank_alert)
        else:
            st.success(user_alert)

        st.session_state.total_txns += 1
        if decision != "ALLOW": st.session_state.total_fraud += 1
        if decision == "BLOCK": st.session_state.total_blocked += 1


# ══════════════════════════════════════════════════════════════
# PAGE 2 — Model Comparison (Paper: LR vs RF vs GB)
# ══════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    st.markdown(
        '<span class="module-badge">Step 4 — ML Model Training Module</span>',
        unsafe_allow_html=True
    )
    st.markdown("Paper finding: Random Forest = most stable | "
                "Gradient Boosting = subtle fraud signals | "
                "Logistic Regression = statistical baseline")
    st.markdown("---")

    if REPORT and "model_comparison" in REPORT:
        mc = REPORT["model_comparison"]
        model_names = list(mc.keys())
        metrics     = ["accuracy","precision","recall","f1","roc_auc"]
        metric_disp = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
        colors      = ["#185FA5","#1D9E75","#E24B4A"]

        # Bar chart comparison
        fig = go.Figure()
        for j, name in enumerate(model_names):
            vals = [mc[name][m] for m in metrics]
            fig.add_trace(go.Bar(
                name=name, x=metric_disp, y=vals,
                marker_color=colors[j], opacity=0.87,
                text=[f"{v:.3f}" for v in vals],
                textposition="outside",
            ))
        fig.update_layout(
            title="Model Performance: Logistic Regression vs Random Forest vs Gradient Boosting",
            barmode="group", yaxis=dict(range=[0,1.15]),
            height=420, legend=dict(orientation="h", y=-0.15)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Metric table
        rows = []
        for name in model_names:
            rows.append({
                "Model"    : name,
                "Accuracy" : mc[name]["accuracy"],
                "Precision": mc[name]["precision"],
                "Recall"   : mc[name]["recall"],
                "F1"       : mc[name]["f1"],
                "ROC-AUC"  : mc[name]["roc_auc"],
                "Paper Note": mc[name].get("note",""),
            })
        df_mc = pd.DataFrame(rows)

        def highlight_best(s):
            is_best = s == s.max()
            return ["background-color:#d4edda;font-weight:600" if v else "" for v in is_best]

        numeric_cols = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
        st.dataframe(
            df_mc.style.apply(highlight_best, subset=numeric_cols),
            use_container_width=True, hide_index=True
        )

        # Show saved plot if available
        base       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        plot_path  = os.path.join(base, "plots", "model_comparison.png")
        if os.path.exists(plot_path):
            st.subheader("Saved Training Plot")
            st.image(plot_path, use_container_width=True)

        cm_path = os.path.join(base, "plots", "confusion_and_pr_curve.png")
        if os.path.exists(cm_path):
            st.subheader("Confusion Matrices + PR Curves (All 3 Models)")
            st.image(cm_path, use_container_width=True)

        best_name = REPORT.get("best_model","")
        if best_name:
            st.success(f"✅ Best model selected for real-time detection: **{best_name}** "
                       f"(Threshold = {REPORT['best_model']['threshold']})")
    else:
        st.info("Run `phase1_train.py` first to see model comparison results.")


# ══════════════════════════════════════════════════════════════
# PAGE 3 — Live Feed
# ══════════════════════════════════════════════════════════════
elif page == "⚡ Live Feed":
    st.title("⚡ Live Transaction Feed")
    st.markdown(
        '<span class="module-badge">Step 5+6 — Real-Time Fraud Detection + Alert System</span>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Monitored",  st.session_state.total_txns)
    m2.metric("Flagged (REVIEW)", st.session_state.total_fraud - st.session_state.total_blocked)
    m3.metric("Blocked",          st.session_state.total_blocked)
    fraud_rate = (st.session_state.total_fraud / st.session_state.total_txns * 100
                  if st.session_state.total_txns > 0 else 0)
    m4.metric("Fraud Rate", f"{fraud_rate:.1f}%")

    st.markdown("")
    c1, c2, c3 = st.columns([1,1,3])
    if c1.button("▶ Start feed",  type="primary"):  st.session_state.live_running = True
    if c2.button("⏹ Stop feed"):                    st.session_state.live_running = False
    inject_fraud = c3.button("⚠️ Inject fraud transaction")

    feed_ph = st.empty()
    status  = st.empty()

    def render_feed():
        if not st.session_state.live_feed:
            feed_ph.info("Click **▶ Start feed** to begin real-time monitoring.")
            return
        rows = st.session_state.live_feed[-30:][::-1]
        df_f = pd.DataFrame(rows)
        def color_rows(row):
            if row["Decision"] == "BLOCK":   return ["background-color:#f8d7da"]*len(row)
            elif row["Decision"] == "REVIEW": return ["background-color:#fff3cd"]*len(row)
            return ["background-color:#d4edda"]*len(row)
        feed_ph.dataframe(df_f.style.apply(color_rows, axis=1),
                          use_container_width=True, height=480)

    def add_txn(fraud=False):
        txn = random_transaction(fraud=fraud)
        _, _, _, prob = predict_all_models(txn)
        decision, risk_level = get_decision(prob)
        factors = get_risk_factors(txn, prob)
        st.session_state.live_feed.append({
            "Time"       : datetime.now().strftime("%H:%M:%S"),
            "Type"       : txn["transaction_type"],
            "City"       : txn["transaction_city"],
            "Amount ₹"   : f"₹{txn['amount_inr']:,.0f}",
            "IP Risk"    : txn["ip_risk_score"],
            "SIM Change" : "Yes" if txn["sim_change_flag"] else "No",
            "Fraud Prob" : f"{prob*100:.1f}%",
            "Decision"   : decision,
            "Alert Sent" : "✓ User+Bank" if decision != "ALLOW" else "—",
            "Flags"      : " | ".join(factors) if factors else "—",
        })
        st.session_state.total_txns += 1
        if decision != "ALLOW": st.session_state.total_fraud += 1
        if decision == "BLOCK": st.session_state.total_blocked += 1

    render_feed()

    if inject_fraud:
        add_txn(fraud=True)
        render_feed()

    if st.session_state.live_running:
        for _ in range(40):
            if not st.session_state.live_running:
                break
            add_txn(fraud=random.random() < 0.08)
            render_feed()
            status.caption(f"Live | {datetime.now().strftime('%H:%M:%S')} | "
                           f"Total: {st.session_state.total_txns} txns")
            time.sleep(0.8)


# ══════════════════════════════════════════════════════════════
# PAGE 4 — Analytics (Step 7: Reporting & Monitoring)
# ══════════════════════════════════════════════════════════════
elif page == "📈 Analytics":
    st.title("📈 Fraud Analytics Dashboard")
    st.markdown(
        '<span class="module-badge">Step 7 — Reporting & Monitoring Module</span>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    @st.cache_data
    def load_data():
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "data", "upi_fraud_dataset_105k.csv")
        return pd.read_csv(path)

    try:
        df = load_data()
    except FileNotFoundError:
        st.error("Dataset not found.")
        st.stop()

    fraud_df = df[df["is_fraud"] == 1]
    legit_df = df[df["is_fraud"] == 0]

    # KPIs
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Transactions",  f"{len(df):,}")
    k2.metric("Fraud Cases",          f"{len(fraud_df):,}")
    k3.metric("Fraud Rate",           f"{len(fraud_df)/len(df)*100:.2f}%")
    k4.metric("Avg Fraud IP Risk",    f"{fraud_df['ip_risk_score'].mean():.3f}")
    st.markdown("---")

    # Row 1: Fraud by hour + by city
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.subheader("Fraud by hour of day")
        hourly = fraud_df.groupby("hour_of_day").size().reset_index(name="count")
        all_h  = pd.DataFrame({"hour_of_day": range(24)})
        hourly = all_h.merge(hourly, on="hour_of_day", how="left").fillna(0)
        fig_h  = px.bar(hourly, x="hour_of_day", y="count",
                        color="count", color_continuous_scale="Reds",
                        labels={"hour_of_day":"Hour","count":"Fraud count"})
        fig_h.update_layout(height=270, margin=dict(t=10,b=10),
                             coloraxis_showscale=False)
        st.plotly_chart(fig_h, use_container_width=True)

    with r1c2:
        st.subheader("Fraud by city (top 10)")
        city_f = (fraud_df.groupby("transaction_city").size()
                  .reset_index(name="count")
                  .sort_values("count", ascending=True).tail(10))
        fig_c = px.bar(city_f, x="count", y="transaction_city",
                       orientation="h", color="count",
                       color_continuous_scale="Oranges",
                       labels={"count":"Fraud cases","transaction_city":"City"})
        fig_c.update_layout(height=270, margin=dict(t=10,b=10),
                             coloraxis_showscale=False)
        st.plotly_chart(fig_c, use_container_width=True)

    # Row 2: Fraud type + transaction type
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.subheader("Fraud type breakdown")
        ftype = (fraud_df.groupby("fraud_type").size()
                 .reset_index(name="count").sort_values("count", ascending=False))
        fig_ft = px.pie(ftype, names="fraud_type", values="count",
                        color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
        fig_ft.update_layout(height=300, margin=dict(t=10,b=10))
        fig_ft.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_ft, use_container_width=True)

    with r2c2:
        st.subheader("Fraud rate by transaction type")
        tx_stats = df.groupby("transaction_type").agg(
            total=("is_fraud","count"), fraud=("is_fraud","sum")
        ).reset_index()
        tx_stats["fraud_rate"] = tx_stats["fraud"] / tx_stats["total"] * 100
        fig_tx = px.bar(tx_stats, x="transaction_type", y="fraud_rate",
                        color="fraud_rate", color_continuous_scale="RdYlGn_r",
                        text=tx_stats["fraud_rate"].round(2).astype(str)+"%",
                        labels={"transaction_type":"Type","fraud_rate":"Fraud rate %"})
        fig_tx.update_layout(height=300, margin=dict(t=10,b=10),
                              coloraxis_showscale=False)
        fig_tx.update_traces(textposition="outside")
        st.plotly_chart(fig_tx, use_container_width=True)

    # Row 3: IP risk dist + PIN attempts
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        st.subheader("IP risk score: Fraud vs Legit")
        sf = fraud_df["ip_risk_score"].sample(min(3000,len(fraud_df)), random_state=42)
        sl = legit_df["ip_risk_score"].sample(min(3000,len(legit_df)), random_state=42)
        fig_ip = go.Figure()
        fig_ip.add_trace(go.Histogram(x=sl, name="Legit",  nbinsx=40,
                                       opacity=0.6, marker_color="#185FA5"))
        fig_ip.add_trace(go.Histogram(x=sf, name="Fraud",  nbinsx=40,
                                       opacity=0.6, marker_color="#E24B4A"))
        fig_ip.update_layout(barmode="overlay", height=270, margin=dict(t=10,b=10),
                              xaxis_title="IP risk score", yaxis_title="Count")
        st.plotly_chart(fig_ip, use_container_width=True)

    with r3c2:
        st.subheader("PIN attempts: Fraud vs Legit")
        pin_stats = df.groupby(["pin_attempts","is_fraud"]).size().reset_index(name="count")
        pin_stats["label"] = pin_stats["is_fraud"].map({0:"Legit",1:"Fraud"})
        fig_pin = px.bar(pin_stats, x="pin_attempts", y="count", color="label",
                         barmode="group",
                         color_discrete_map={"Legit":"#185FA5","Fraud":"#E24B4A"},
                         labels={"pin_attempts":"PIN attempts","count":"Count","label":""})
        fig_pin.update_layout(height=270, margin=dict(t=10,b=10))
        st.plotly_chart(fig_pin, use_container_width=True)

    # SIM + Device heatmap
    st.subheader("Fraud rate: SIM Change × New Device")
    heat = df.groupby(["sim_change_flag","is_new_device"]).agg(
        fr=("is_fraud","mean")).reset_index()
    heat_p = heat.pivot(index="sim_change_flag",
                        columns="is_new_device", values="fr")
    heat_p.index   = ["SIM unchanged","SIM changed"]
    heat_p.columns = ["Known device","New device"]
    fig_h2 = px.imshow(heat_p*100, text_auto=".2f",
                        color_continuous_scale="Reds",
                        labels={"color":"Fraud rate %"}, aspect="auto")
    fig_h2.update_layout(height=200, margin=dict(t=10,b=10))
    st.plotly_chart(fig_h2, use_container_width=True)

    st.caption(f"Dataset: {len(df):,} transactions | Fraud: {len(fraud_df):,} | "
               f"Threshold: {THRESHOLD:.3f} | Best model: {BEST_MODEL_NAME}")


# ══════════════════════════════════════════════════════════════
# PAGE 5 — System Pipeline (Research Paper Architecture)
# ══════════════════════════════════════════════════════════════
elif page == "🗺️ System Pipeline":
    st.title("🗺️ System Pipeline")
    st.markdown("Research paper architecture — all 7 modules")
    st.markdown("---")

    pipeline_modules = [
        ("1", "Data Collection Module",
         "Collects transaction amount, timestamp, device ID, geographical location, "
         "number of transactions in time period, user behavioral history.",
         "🗄️", "#e3f2fd"),
        ("2", "Data Preprocessing Module",
         "Removes duplicates, handles missing values, encodes categorical features "
         "(transaction type, city, merchant category), applies StandardScaler normalization.",
         "⚙️", "#e8f5e9"),
        ("3", "Feature Extraction Module",
         "Extracts behavioral indicators: spend patterns, transaction frequency, "
         "location & device anomalies. Red flags: high velocity, new device, odd hour, "
         "large amount deviation.",
         "🔬", "#fff3e0"),
        ("4", "ML Model Training Module",
         "Three supervised models trained on historical data:\n"
         "• Logistic Regression — basic statistical classification\n"
         "• Random Forest — stable, combines multiple behavioral features\n"
         "• Gradient Boosting — detects subtle fraudulent signals",
         "🤖", "#f3e5f5"),
        ("5", "Fraud Detection Module",
         "Trained model generates a fraud risk score (0–100) per transaction. "
         "Score compared against tuned threshold → signed as ALLOW / REVIEW / BLOCK.",
         "🎯", "#fce4ec"),
        ("6", "Alert & Notification Module",
         "On suspicious detection: immediately notifies USER (to verify) "
         "AND BANKING AUTHORITY (to block or add authentication). "
         "Prevents financial loss, increases user awareness.",
         "🔔", "#fff8e1"),
        ("7", "Reporting & Monitoring Module",
         "All transactions + predictions stored in database. "
         "Fraud trend reports generated for bank administrators. "
         "Stored data used for periodic model retraining.",
         "📊", "#e0f7fa"),
    ]

    for num, title, desc, icon, color in pipeline_modules:
        with st.container():
            st.markdown(
                f'<div style="background:{color};border-radius:10px;padding:14px 18px;'
                f'margin-bottom:10px;border-left:4px solid #1565c0">'
                f'<b>{icon} Step {num}: {title}</b><br>'
                f'<span style="font-size:13px;color:#333">{desc.replace(chr(10),"<br>")}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.subheader("System Architecture Components")
    arch_items = [
        "User Mobile Application",
        "UPI Transaction Server",
        "Data Preprocessing Module",
        "Feature Extraction Module",
        "Machine Learning Prediction Engine (LR / RF / GB)",
        "Fraud Decision Module (Risk Score Threshold)",
        "Bank Processing Unit",
        "Alert & Notification System (User + Bank)",
        "Database Storage (Transaction Logs + Reports)",
    ]
    for i, item in enumerate(arch_items, 1):
        st.markdown(f"**{i}.** {item}")

    if REPORT:
        st.markdown("---")
        st.subheader("Last Training Run Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Dataset size",   f"{REPORT.get('dataset',{}).get('size',0):,}")
        col2.metric("Best model",      REPORT.get("best_model",{}).get("name","—"))
        col3.metric("Best ROC-AUC",   REPORT.get("best_model",{}).get("roc_auc","—"))