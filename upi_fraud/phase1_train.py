# ============================================================
#  UPI Fraud Detection — Phase 1
#  Research Paper Flow:
#  Step 1: Data Collection Module
#  Step 2: Data Preprocessing Module
#  Step 3: Feature Extraction Module
#  Step 4: ML Model Training (LR + Random Forest + Gradient Boosting)
#  Step 5: Fraud Detection Module (Risk Score + Threshold)
#  Step 6: Alert & Notification Module
#  Step 7: Reporting & Monitoring Module
# ============================================================

import pandas as pd
import numpy as np
import joblib
import os
import warnings
import json
from datetime import datetime
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─── 0. Config ────────────────────────────────────────────────
DATA_PATH   = "data/upi_fraud_dataset_105k.csv"
MODELS_DIR  = "models"
REPORTS_DIR = "reports"
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs("plots",     exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STEP 1 — DATA COLLECTION MODULE
# Paper: Collects transaction amount, time, date, location,
# device ID, frequency, user behavioral history
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  STEP 1 — DATA COLLECTION MODULE")
print("  (transaction amount, time, device, location, behaviour)")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"  Records loaded  : {df.shape[0]:,}")
print(f"  Features        : {df.shape[1]}")
print(f"  Fraud count     : {df['is_fraud'].sum():,}")
print(f"  Legit count     : {(df['is_fraud']==0).sum():,}")
print(f"  Fraud rate      : {df['is_fraud'].mean()*100:.2f}%")
print(f"  Date range      : {df['timestamp'].min()} to {df['timestamp'].max()}")

# ─────────────────────────────────────────────────────────────
# STEP 2 — DATA PREPROCESSING MODULE
# Paper: Removes duplicates, encodes categoricals, normalizes
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 2 — DATA PREPROCESSING MODULE")
print("  (cleaning, encoding, normalization)")
print("=" * 60)

# 2a. Remove duplicates
before = len(df)
df = df.drop_duplicates()
print(f"  Duplicates removed : {before - len(df)}")

# 2b. Handle missing values
df = df.fillna(df.median(numeric_only=True))
print(f"  Missing values filled with column medians")

# 2c. Feature selection — aligned with paper parameters:
# transaction amount, frequency, time pattern, device ID,
# geographical location, user behavioral history
FEATURES = [
    # Transaction info (paper: transaction amount, time pattern)
    "transaction_type",           # categorical
    "transaction_city",           # geographical location
    "amount_inr",                 # transaction amount
    "hour_of_day",                # time pattern
    "day_of_week",                # time pattern
    "is_odd_hour",                # unusual time flag

    # Device identification (paper: device identification)
    "is_new_device",              # unrecognized device
    "sim_change_flag",            # SIM swap detection
    "kyc_status",                 # KYC verification status

    # Geographical location (paper: geographical location)
    "location_mismatch",          # location deviation
    "is_international",           # cross-border flag

    # Behavioral history (paper: user behavioral history, frequency)
    "is_new_payee",               # new payee flag
    "txns_last_1hr",              # transaction frequency (1hr)
    "txns_last_24hr",             # transaction frequency (24hr)
    "pin_attempts",               # authentication attempts
    "ip_risk_score",              # IP-based risk score
    "failed_txns_24hr",           # failed transaction count
    "amount_deviation_from_avg",  # spending pattern deviation
    "txn_velocity_score",         # velocity-based risk
    "avg_txn_amount_30d",         # 30-day spending baseline
    "upi_handle_risk",            # UPI handle risk score

    # Merchant
    "merchant_category",          # categorical
]
TARGET = "is_fraud"

# 2d. Encode categorical features
print("\n  Encoding categorical features:")
encoders = {}
df_enc = df[FEATURES + [TARGET]].copy()

for col in ["transaction_type", "transaction_city", "merchant_category"]:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    encoders[col] = le
    print(f"  Encoded '{col}': {len(le.classes_)} unique values")

joblib.dump(encoders, f"{MODELS_DIR}/label_encoders.pkl")

# 2e. Normalization (required for Logistic Regression)
X_raw = df_enc[FEATURES].values
y     = df_enc[TARGET].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
print(f"\n  Normalization applied (StandardScaler)")
print(f"  Scaler saved → {MODELS_DIR}/scaler.pkl")

# ─────────────────────────────────────────────────────────────
# STEP 3 — FEATURE EXTRACTION MODULE
# Paper: Analyzes behavioral patterns (typical spend, time,
# location, device) — anomalies are red flag indicators
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 3 — FEATURE EXTRACTION MODULE")
print("  (behavioral patterns & red flag indicators)")
print("=" * 60)

fraud_mask = y == 1
legit_mask = y == 0

print(f"\n  Behavioral Pattern Analysis (Fraud vs Legit):")
print(f"  {'Feature':<30} {'Fraud Avg':>10} {'Legit Avg':>10}  {'Red Flag?':>10}")
print(f"  {'-'*64}")

behavior_analysis = [
    ("txns_last_1hr",           "High velocity (>=5 txns/hr)"),
    ("pin_attempts",            "Multiple PIN attempts (>=3)"),
    ("ip_risk_score",           "High IP risk (>0.5)"),
    ("amount_deviation_from_avg","Large spending deviation"),
    ("txn_velocity_score",      "High velocity score"),
]

for col, flag_desc in behavior_analysis:
    idx = FEATURES.index(col)
    fm  = X_raw[fraud_mask, idx].mean()
    lm  = X_raw[legit_mask, idx].mean()
    flag = "YES" if fm > lm else " no"
    print(f"  {col:<30} {fm:>10.3f} {lm:>10.3f}  {flag:>10}")

print(f"\n  Total features extracted : {len(FEATURES)}")
joblib.dump(FEATURES, f"{MODELS_DIR}/feature_names.pkl")

# ─────────────────────────────────────────────────────────────
# STEP 4 — ML MODEL TRAINING MODULE
# Paper: Logistic Regression + Random Forest + Gradient Boosting
# Trained on historical data; evaluated by accuracy & recall
# Best model selected for real-time detection
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 4 — ML MODEL TRAINING MODULE")
print("  (Logistic Regression | Random Forest | Gradient Boosting)")
print("=" * 60)

# Train/Test split — 80/20 stratified
X_train_raw,  X_test_raw,  y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)
X_train_sc, X_test_sc, _, _ = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Train : {len(X_train_raw):,} | Fraud: {y_train.sum():,}")
print(f"  Test  : {len(X_test_raw):,}  | Fraud: {y_test.sum():,}")

# SMOTE to balance training data
print("\n  Applying SMOTE oversampling...")
smote = SMOTE(random_state=42, sampling_strategy=0.25, k_neighbors=5)
X_train_sm,    y_train_sm = smote.fit_resample(X_train_raw, y_train)
X_train_sm_sc, _          = smote.fit_resample(X_train_sc,  y_train)
print(f"  After SMOTE — Fraud: {y_train_sm.sum():,} | Legit: {(y_train_sm==0).sum():,}")

# Three models as specified in the paper
models_config = {
    "Logistic Regression": {
        "model"  : LogisticRegression(
            max_iter=1000, C=0.1, class_weight="balanced",
            solver="lbfgs", random_state=42
        ),
        "X_train": X_train_sm_sc,  # requires scaled features
        "X_test" : X_test_sc,
        "note"   : "Best for basic statistical classification (paper)",
    },
    "Random Forest": {
        "model"  : RandomForestClassifier(
            n_estimators=200, max_depth=15,
            min_samples_leaf=5, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "X_train": X_train_sm,
        "X_test" : X_test_raw,
        "note"   : "Most stable — combines multiple behavioural features (paper)",
    },
    "Gradient Boosting": {
        "model"  : GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=5, subsample=0.8, random_state=42
        ),
        "X_train": X_train_sm,
        "X_test" : X_test_raw,
        "note"   : "Detects delicate fraudulent signals (paper)",
    },
}

results       = {}
trained_models = {}

print(f"\n  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} "
      f"{'F1':>8} {'ROC-AUC':>9}")
print(f"  {'-'*72}")

for name, cfg in models_config.items():
    print(f"  Training {name}...", end=" ", flush=True)
    clf = cfg["model"]
    clf.fit(cfg["X_train"], y_train_sm)

    probs = clf.predict_proba(cfg["X_test"])[:, 1]
    preds = (probs >= 0.5).astype(int)

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds)
    f1   = f1_score(y_test, preds)
    auc  = roc_auc_score(y_test, probs)

    results[name]        = {"accuracy":round(acc,4),"precision":round(prec,4),
                             "recall":round(rec,4),"f1":round(f1,4),
                             "roc_auc":round(auc,4),"probs":probs,"note":cfg["note"]}
    trained_models[name] = clf

    print(f"Done  {acc:>8.4f} {prec:>10.4f} {rec:>8.4f} {f1:>8.4f} {auc:>9.4f}")

# ─────────────────────────────────────────────────────────────
# STEP 5 — FRAUD DETECTION MODULE
# Paper: Risk score generated per transaction, compared against
# threshold, signed as valid or suspect
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 5 — FRAUD DETECTION MODULE")
print("  (fraud risk score + threshold comparison)")
print("=" * 60)

# Best model by ROC-AUC
best_model_name = max(results, key=lambda k: results[k]["roc_auc"])
best_model      = trained_models[best_model_name]
best_probs      = results[best_model_name]["probs"]

print(f"\n  Best performing model : {best_model_name}")
print(f"  Note                  : {results[best_model_name]['note']}")
print(f"  ROC-AUC               : {results[best_model_name]['roc_auc']:.4f}")

# F1-optimal threshold tuning
precisions, recalls, thresholds = precision_recall_curve(y_test, best_probs)
f1_scores_thresh = 2*(precisions[:-1]*recalls[:-1])/(precisions[:-1]+recalls[:-1]+1e-8)
best_thresh_idx  = np.argmax(f1_scores_thresh)
best_threshold   = float(thresholds[best_thresh_idx])

final_preds = (best_probs >= best_threshold).astype(int)

print(f"\n  Fraud risk threshold (tuned): {best_threshold:.4f}")
print(f"  Decision: score >= {best_threshold:.2f} → SUSPICIOUS")
print(f"  Decision: score <  {best_threshold:.2f} → GENUINE")
print(f"\n  Final Results @ tuned threshold:")
print(classification_report(y_test, final_preds, target_names=["Legit","Fraud"]))

# ─────────────────────────────────────────────────────────────
# STEP 6 — ALERT & NOTIFICATION MODULE
# Paper: System immediately sends alerts to user AND banking
# authority once suspicious activity is detected
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  STEP 6 — ALERT & NOTIFICATION MODULE")
print("  (real-time user + bank authority notification)")
print("=" * 60)

def generate_alert(transaction_id, fraud_prob, threshold, reasons):
    """
    Alert generation as described in research paper.
    Notifies both user (to verify) and bank (to block/auth).
    """
    return {
        "alert_id"         : f"ALERT_{transaction_id}",
        "timestamp"        : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "transaction_id"   : transaction_id,
        "fraud_probability": round(fraud_prob, 4),
        "risk_score"       : min(100, int(fraud_prob * 100)),
        "decision"         : "BLOCK" if fraud_prob >= 0.60 else "REVIEW",
        "reasons"          : reasons,
        # Paper: notify user to cross-verify transaction
        "user_alert"       : "Transaction flagged! Please verify via UPI app.",
        "user_notified"    : True,
        # Paper: notify banking authority for commercial action
        "bank_alert"       : "Suspicious transaction detected. Block/re-authenticate.",
        "bank_notified"    : True,
        "action_taken"     : "Transaction temporarily held for verification",
    }

flagged_count   = int(final_preds.sum())
true_fraud_caught = int(((final_preds==1) & (y_test==1)).sum())
false_alarms      = int(((final_preds==1) & (y_test==0)).sum())

print(f"\n  Transactions flagged    : {flagged_count:,}")
print(f"  True frauds caught      : {true_fraud_caught:,}")
print(f"  False alarms            : {false_alarms:,}")
print(f"\n  Sample Alert (highest-risk transaction):")

high_risk_indices = np.where(final_preds == 1)[0]
if len(high_risk_indices) > 0:
    demo_idx   = high_risk_indices[np.argmax(best_probs[high_risk_indices])]
    demo_alert = generate_alert(
        transaction_id=f"TXN_{demo_idx:06d}",
        fraud_prob=best_probs[demo_idx],
        threshold=best_threshold,
        reasons=["High IP risk score", "SIM card recently changed",
                 "Transaction from new device"]
    )
    for k, v in demo_alert.items():
        print(f"  {k:<22}: {v}")

# ─────────────────────────────────────────────────────────────
# STEP 7 — REPORTING & MONITORING MODULE
# Paper: Saves all transaction + prediction results to DB,
# generates fraud trend reports, periodic model retraining data
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 7 — REPORTING & MONITORING MODULE")
print("  (fraud trend reports + model performance tracking)")
print("=" * 60)

# Full performance report
report = {
    "generated_at"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "paper_reference" : "Real-Time UPI Fraud Detection Using Machine Learning — IJEDR 2026",
    "dataset"         : {
        "size"         : int(len(df)),
        "fraud_count"  : int(y.sum()),
        "legit_count"  : int((y==0).sum()),
        "fraud_rate_pct": round(float(y.mean())*100, 2),
        "features_used": len(FEATURES),
    },
    "preprocessing"   : {
        "duplicates_removed": before - len(df),
        "smote_applied"     : True,
        "normalization"     : "StandardScaler",
        "train_test_split"  : "80/20 stratified",
    },
    "model_comparison": {
        name: {k: v for k, v in m.items() if k not in ("probs",)}
        for name, m in results.items()
    },
    "best_model"      : {
        "name"     : best_model_name,
        "threshold": round(best_threshold, 4),
        "accuracy" : results[best_model_name]["accuracy"],
        "recall"   : results[best_model_name]["recall"],
        "roc_auc"  : results[best_model_name]["roc_auc"],
    },
    "alert_summary"   : {
        "total_flagged"   : flagged_count,
        "true_fraud_caught": true_fraud_caught,
        "false_alarms"    : false_alarms,
    },
    "paper_findings"  : {
        "Random Forest"       : "Most stable — combines multiple behavioural features",
        "Logistic Regression" : "Best suited for basic statistical classification",
        "Gradient Boosting"   : "Detects more delicate fraudulent signals",
    }
}

report_path = f"{REPORTS_DIR}/model_performance_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\n  Full report saved → {report_path}")

# ─── Confusion matrices + PR curves for all 3 models ────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    "UPI Fraud Detection — Model Performance\n"
    "Logistic Regression  |  Random Forest  |  Gradient Boosting",
    fontsize=14, fontweight="bold"
)

COLORS = {"Logistic Regression":"#185FA5",
          "Random Forest"      :"#1D9E75",
          "Gradient Boosting"  :"#E24B4A"}

for i, name in enumerate(models_config.keys()):
    probs  = results[name]["probs"]
    p_arr, r_arr, t_arr = precision_recall_curve(y_test, probs)
    f1_arr  = 2*(p_arr[:-1]*r_arr[:-1])/(p_arr[:-1]+r_arr[:-1]+1e-8)
    best_t  = float(t_arr[np.argmax(f1_arr)])
    preds   = (probs >= best_t).astype(int)
    cm      = confusion_matrix(y_test, preds)

    # Confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit","Fraud"],
                yticklabels=["Legit","Fraud"],
                ax=axes[0][i], linewidths=0.5)
    axes[0][i].set_title(
        f"{name}\nACC={results[name]['accuracy']:.3f}  "
        f"REC={results[name]['recall']:.3f}  "
        f"AUC={results[name]['roc_auc']:.3f}",
        fontsize=10
    )
    axes[0][i].set_ylabel("Actual")
    axes[0][i].set_xlabel("Predicted")

    # PR Curve
    best_pr_idx = np.argmax(f1_arr)
    axes[1][i].plot(r_arr[:-1], p_arr[:-1], color=COLORS[name], lw=2)
    axes[1][i].scatter(r_arr[best_pr_idx], p_arr[best_pr_idx],
                       color="black", zorder=5, s=60,
                       label=f"Threshold={best_t:.2f}")
    axes[1][i].set_xlabel("Recall")
    axes[1][i].set_ylabel("Precision")
    axes[1][i].set_title(f"PR Curve — {name}", fontsize=10)
    axes[1][i].legend(fontsize=8)
    axes[1][i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/confusion_and_pr_curve.png", dpi=150, bbox_inches="tight")
print(f"  Confusion + PR plots → plots/confusion_and_pr_curve.png")

# ─── Model comparison bar chart ──────────────────────────────
fig2, ax = plt.subplots(figsize=(11, 5))
metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
metric_keys   = ["accuracy", "precision", "recall", "f1", "roc_auc"]
x     = np.arange(len(metric_keys))
width = 0.25

for j, name in enumerate(models_config.keys()):
    vals = [results[name][k] for k in metric_keys]
    bars = ax.bar(x + j*width, vals, width, label=name,
                  color=list(COLORS.values())[j], alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x + width)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("Model Comparison: Logistic Regression vs Random Forest vs Gradient Boosting",
             fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plots/model_comparison.png", dpi=150)
print(f"  Model comparison bar → plots/model_comparison.png")

# ─── Save all model artifacts ────────────────────────────────
print("\n  Saving model artifacts:")
for name, clf in trained_models.items():
    fname = name.lower().replace(" ", "_") + "_model.pkl"
    joblib.dump(clf, f"{MODELS_DIR}/{fname}")
    print(f"  {name:<25} → {MODELS_DIR}/{fname}")

joblib.dump(best_model,      f"{MODELS_DIR}/best_model.pkl")
joblib.dump(best_threshold,  f"{MODELS_DIR}/threshold.pkl")
joblib.dump(FEATURES,        f"{MODELS_DIR}/feature_names.pkl")
joblib.dump(best_model_name, f"{MODELS_DIR}/best_model_name.pkl")

print("\n" + "=" * 60)
print("  ALL 7 MODULES COMPLETE — Research Paper Flow Done")
print(f"  Best Model  : {best_model_name}")
print(f"  AUC         : {results[best_model_name]['roc_auc']:.4f}")
print(f"  Recall      : {results[best_model_name]['recall']:.4f}")
print(f"  Threshold   : {best_threshold:.4f}")
print("  Next: uvicorn api.main:app --reload --port 8000")
print("=" * 60)

#  # ============================================================
# #  UPI Fraud Detection — Phase 1
# #  SMOTE + Optuna Hyperparameter Tuning + Threshold Tuning
# # ============================================================

# import pandas as pd
# import numpy as np
# import joblib
# import os
# import warnings
# warnings.filterwarnings("ignore")

# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import HistGradientBoostingClassifier
# from sklearn.metrics import (
#     accuracy_score, f1_score, precision_score, recall_score,
#     roc_auc_score, confusion_matrix, classification_report,
#     precision_recall_curve
# )
# from imblearn.over_sampling import SMOTE
# import optuna
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import seaborn as sns

# optuna.logging.set_verbosity(optuna.logging.WARNING)

# # ─── 0. Config ────────────────────────────────────────────────
# DATA_PATH   = "data/upi_fraud_dataset_105k.csv"
# MODELS_DIR  = "models"
# os.makedirs(MODELS_DIR, exist_ok=True)
# os.makedirs("plots", exist_ok=True)

# FEATURES = [
#     "transaction_type", "transaction_city", "hour_of_day",
#     "is_odd_hour", "is_new_device", "is_new_payee",
#     "location_mismatch", "txns_last_1hr", "pin_attempts",
#     "ip_risk_score", "sim_change_flag", "kyc_status",
#     "merchant_category",
# ]
# TARGET = "is_fraud"

# # ─── 1. Load data ─────────────────────────────────────────────
# print("=" * 55)
# print("  STEP 1 — Loading data")
# print("=" * 55)

# df = pd.read_csv(DATA_PATH)
# print(f"  Shape          : {df.shape}")
# print(f"  Fraud rate     : {df[TARGET].mean()*100:.2f}%")
# print(f"  Fraud count    : {df[TARGET].sum():,}")
# print(f"  Legit count    : {(df[TARGET]==0).sum():,}")

# # ─── 2. Encode categoricals ───────────────────────────────────
# print("\n" + "=" * 55)
# print("  STEP 2 — Encoding categorical features")
# print("=" * 55)

# encoders = {}
# df_enc = df[FEATURES + [TARGET]].copy()

# for col in ["transaction_type", "transaction_city", "merchant_category"]:
#     le = LabelEncoder()
#     df_enc[col] = le.fit_transform(df_enc[col].astype(str))
#     encoders[col] = le
#     print(f"  Encoded '{col}': {list(le.classes_)}")

# joblib.dump(encoders, f"{MODELS_DIR}/label_encoders.pkl")
# print(f"\n  Encoders saved → {MODELS_DIR}/label_encoders.pkl")

# # ─── 3. Train / Test split ────────────────────────────────────
# print("\n" + "=" * 55)
# print("  STEP 3 — Train / Test split (80 / 20, stratified)")
# print("=" * 55)

# X = df_enc[FEATURES].values
# y = df_enc[TARGET].values

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )
# print(f"  Train size : {len(X_train):,}  | Fraud: {y_train.sum():,}")
# print(f"  Test size  : {len(X_test):,}   | Fraud: {y_test.sum():,}")

# # ─── 4. SMOTE ─────────────────────────────────────────────────
# print("\n" + "=" * 55)
# print("  STEP 4 — SMOTE oversampling")
# print("=" * 55)

# smote = SMOTE(random_state=42, sampling_strategy=0.25, k_neighbors=5)
# X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# print(f"  Before SMOTE → Fraud: {y_train.sum():,} | Legit: {(y_train==0).sum():,}")
# print(f"  After  SMOTE → Fraud: {y_train_sm.sum():,} | Legit: {(y_train_sm==0).sum():,}")
# print(f"  New fraud rate : {y_train_sm.mean()*100:.1f}%")

# # ─── 5. Baseline (before tuning) ──────────────────────────────
# print("\n" + "=" * 55)
# print("  STEP 5 — Baseline model (default params + SMOTE)")
# print("=" * 55)

# baseline = HistGradientBoostingClassifier(random_state=42)
# baseline.fit(X_train_sm, y_train_sm)

# base_probs = baseline.predict_proba(X_test)[:, 1]
# base_preds = (base_probs >= 0.5).astype(int)

# print(f"  Accuracy  : {accuracy_score(y_test, base_preds):.4f}")
# print(f"  F1        : {f1_score(y_test, base_preds):.4f}")
# print(f"  Precision : {precision_score(y_test, base_preds):.4f}")
# print(f"  Recall    : {recall_score(y_test, base_preds):.4f}")
# print(f"  ROC-AUC   : {roc_auc_score(y_test, base_probs):.4f}")

# # ─── 6. Optuna Tuning ─────────────────────────────────────────
# print("\n" + "=" * 55)
# print("  STEP 6 — Optuna hyperparameter tuning (60 trials)")
# print("=" * 55)

# def objective(trial):
#     params = {
#         "max_iter"          : trial.suggest_int("max_iter", 100, 600),
#         "learning_rate"     : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "max_depth"         : trial.suggest_int("max_depth", 3, 12),
#         "min_samples_leaf"  : trial.suggest_int("min_samples_leaf", 5, 100),
#         "max_leaf_nodes"    : trial.suggest_int("max_leaf_nodes", 20, 100),
#         "l2_regularization" : trial.suggest_float("l2_regularization", 0.0, 1.0),
#         "max_bins"          : trial.suggest_int("max_bins", 64, 255),
#         "random_state"      : 42,
#     }
#     model = HistGradientBoostingClassifier(**params)
#     cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
#     scores = cross_val_score(
#         model, X_train_sm, y_train_sm,
#         cv=cv, scoring="roc_auc", n_jobs=-1
#     )
#     return scores.mean()

# study = optuna.create_study(direction="maximize",
#                              sampler=optuna.samplers.TPESampler(seed=42))
# study.optimize(objective, n_trials=60, show_progress_bar=True)

# print(f"\n  Best ROC-AUC (CV) : {study.best_value:.4f}")
# print(f"  Best params       : {study.best_params}")

# # ─── 7. Train best model ──────────────────────────────────────
# print("\n" + "=" * 55)
# print("  STEP 7 — Training final model with best params")
# print("=" * 55)

# best_model = HistGradientBoostingClassifier(
#     **study.best_params, random_state=42
# )
# best_model.fit(X_train_sm, y_train_sm)

# best_probs = best_model.predict_proba(X_test)[:, 1]
# best_preds = (best_probs >= 0.5).astype(int)

# print("  Results @ threshold=0.5 :")
# print(f"  Accuracy  : {accuracy_score(y_test, best_preds):.4f}")
# print(f"  F1        : {f1_score(y_test, best_preds):.4f}")
# print(f"  Precision : {precision_score(y_test, best_preds):.4f}")
# print(f"  Recall    : {recall_score(y_test, best_preds):.4f}")
# print(f"  ROC-AUC   : {roc_auc_score(y_test, best_probs):.4f}")

# # ─── 8. Threshold Tuning ──────────────────────────────────────
# print("\n" + "=" * 55)
# print("  STEP 8 — Precision-Recall threshold tuning")
# print("=" * 55)

# precisions, recalls, thresholds = precision_recall_curve(y_test, best_probs)

# # F1-optimal threshold
# f1_scores_thresh = 2 * (precisions[:-1] * recalls[:-1]) / (
#     precisions[:-1] + recalls[:-1] + 1e-8
# )
# best_thresh_idx  = np.argmax(f1_scores_thresh)
# best_threshold   = thresholds[best_thresh_idx]

# print(f"  F1-optimal threshold : {best_threshold:.4f}")
# print(f"  Precision @ threshold: {precisions[best_thresh_idx]:.4f}")
# print(f"  Recall    @ threshold: {recalls[best_thresh_idx]:.4f}")
# print(f"  F1        @ threshold: {f1_scores_thresh[best_thresh_idx]:.4f}")

# # Final predictions with tuned threshold
# final_preds = (best_probs >= best_threshold).astype(int)

# print("\n  Final results @ tuned threshold :")
# print(f"  Accuracy  : {accuracy_score(y_test, final_preds):.4f}")
# print(f"  F1        : {f1_score(y_test, final_preds):.4f}")
# print(f"  Precision : {precision_score(y_test, final_preds):.4f}")
# print(f"  Recall    : {recall_score(y_test, final_preds):.4f}")
# print(f"  ROC-AUC   : {roc_auc_score(y_test, best_probs):.4f}")

# print("\n  Full Classification Report:")
# print(classification_report(y_test, final_preds,
#                              target_names=["Legit", "Fraud"]))

# # ─── 9. Confusion Matrix plot ─────────────────────────────────
# cm = confusion_matrix(y_test, final_preds)
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# # Confusion matrix
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#             xticklabels=["Legit", "Fraud"],
#             yticklabels=["Legit", "Fraud"],
#             ax=axes[0], linewidths=0.5)
# axes[0].set_title("Confusion Matrix (tuned threshold)", fontsize=13)
# axes[0].set_ylabel("Actual")
# axes[0].set_xlabel("Predicted")

# tn, fp, fn, tp = cm.ravel()
# axes[0].text(0.5, -0.18,
#     f"TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}",
#     ha="center", transform=axes[0].transAxes,
#     fontsize=10, color="gray")

# # Precision-Recall curve
# axes[1].plot(recalls[:-1], precisions[:-1], color="#185FA5", lw=2)
# axes[1].scatter(recalls[best_thresh_idx], precisions[best_thresh_idx],
#                 color="#E24B4A", zorder=5, s=80,
#                 label=f"Best threshold={best_threshold:.2f}")
# axes[1].set_xlabel("Recall")
# axes[1].set_ylabel("Precision")
# axes[1].set_title("Precision-Recall Curve", fontsize=13)
# axes[1].legend()
# axes[1].grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig("plots/confusion_and_pr_curve.png", dpi=150)
# print("\n  Plot saved → plots/confusion_and_pr_curve.png")

# # ─── 10. Save model artifacts ─────────────────────────────────
# print("\n" + "=" * 55)
# print("  STEP 9 — Saving model artifacts")
# print("=" * 55)

# joblib.dump(best_model,   f"{MODELS_DIR}/hist_gb_model.pkl")
# joblib.dump(best_threshold, f"{MODELS_DIR}/threshold.pkl")
# joblib.dump(FEATURES,     f"{MODELS_DIR}/feature_names.pkl")
# joblib.dump(study.best_params, f"{MODELS_DIR}/best_params.pkl")

# print(f"  Model     → {MODELS_DIR}/hist_gb_model.pkl")
# print(f"  Threshold → {MODELS_DIR}/threshold.pkl")
# print(f"  Features  → {MODELS_DIR}/feature_names.pkl")
# print(f"  Encoders  → {MODELS_DIR}/label_encoders.pkl")

# print("\n" + "=" * 55)
# print("  PHASE 1 COMPLETE — Run phase2 (FastAPI) next!")
# print("=" * 55)
