# ============================================================
#  UPI Fraud Detection — FastAPI Backend
#  Research Paper Flow (IJEDR 2026):
#
#  Every /predict call executes the paper's pipeline:
#  Step 1: Data Collection (capture transaction parameters)
#  Step 2: Data Preprocessing (clean, encode, normalize)
#  Step 3: Feature Extraction (behavioral indicators)
#  Step 4: ML Prediction (best of LR/RF/GB ensemble)
#  Step 5: Fraud Detection (risk score vs threshold)
#  Step 6: Alert & Notification (user + bank alerts)
#  Step 7: Reporting & Monitoring (log to DB)
#
#  Run: uvicorn api.main:app --reload --port 8000
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import joblib
import numpy as np
import os
import time
import json
from datetime import datetime
from collections import defaultdict

# ─── Load model artifacts ─────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

try:
    # Load best model (Random Forest / Gradient Boosting — paper recommended)
    best_model      = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    best_model_name = joblib.load(os.path.join(MODELS_DIR, "best_model_name.pkl"))
    threshold       = joblib.load(os.path.join(MODELS_DIR, "threshold.pkl"))
    encoders        = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
    features        = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    scaler          = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

    # Also load all three models for /compare endpoint
    lr_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression_model.pkl"))
    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.pkl"))
    gb_model = joblib.load(os.path.join(MODELS_DIR, "gradient_boosting_model.pkl"))

    print(f"[OK] Models loaded | Best: {best_model_name} | Threshold: {threshold:.4f}")
except FileNotFoundError as e:
    raise RuntimeError(f"Model files not found. Run phase1_train.py first.\n{e}")

# ─── FastAPI app ──────────────────────────────────────────────
app = FastAPI(
    title="UPI Fraud Detection API",
    description=(
        "Real-time UPI transaction fraud detection system based on "
        "research paper: 'Real-Time UPI Fraud Detection Using Machine Learning' "
        "(IJEDR 2026). Implements all 7 modules of the paper's pipeline."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory transaction log (Step 7: Monitoring) ──────────
transaction_log: list = []
alert_log:       list = []
START_TIME = time.time()
tx_counter = 0

# ─── Request Schema ───────────────────────────────────────────
class TransactionRequest(BaseModel):
    """
    Paper Step 1 — Data Collection Module parameters:
    transaction amount, time, device ID, location, behavioral history
    """
    # Transaction info
    transaction_type          : str   = Field(..., example="P2P",
        description="P2P / P2M / Bill / Recharge")
    transaction_city          : str   = Field(..., example="Delhi")
    amount_inr                : float = Field(..., gt=0, example=5000.0,
        description="Transaction amount in INR")
    hour_of_day               : int   = Field(..., ge=0, le=23, example=14)
    day_of_week               : int   = Field(..., ge=0, le=6,  example=2)
    is_odd_hour               : int   = Field(..., ge=0, le=1,  example=0,
        description="1 if transaction is between 11PM-6AM")

    # Device identification (paper: device ID)
    is_new_device             : int   = Field(..., ge=0, le=1,  example=0)
    sim_change_flag           : int   = Field(..., ge=0, le=1,  example=0)
    kyc_status                : int   = Field(..., ge=0, le=2,  example=1,
        description="0=Pending, 1=Verified, 2=Rejected")

    # Geographical location (paper: geographical location)
    location_mismatch         : int   = Field(..., ge=0, le=1,  example=0)
    is_international          : int   = Field(..., ge=0, le=1,  example=0)

    # Behavioral history (paper: user behavioral history)
    is_new_payee              : int   = Field(..., ge=0, le=1,  example=0)
    txns_last_1hr             : int   = Field(..., ge=0,         example=1)
    txns_last_24hr            : int   = Field(..., ge=0,         example=5)
    pin_attempts              : int   = Field(..., ge=1, le=3,  example=1)
    ip_risk_score             : float = Field(..., ge=0, le=1,  example=0.1)
    failed_txns_24hr          : int   = Field(..., ge=0,         example=0)
    amount_deviation_from_avg : float = Field(...,               example=0.5)
    txn_velocity_score        : float = Field(..., ge=0, le=1,  example=0.1)
    avg_txn_amount_30d        : float = Field(..., ge=0,         example=3000.0)
    upi_handle_risk           : float = Field(..., ge=0, le=1,  example=0.05)

    # Merchant
    merchant_category         : str   = Field(..., example="food",
        description="food/grocery/travel/entertainment/utility/retail/healthcare/peer")

    @validator("transaction_type")
    def valid_tx_type(cls, v):
        allowed = ["P2P", "P2M", "Bill", "Recharge"]
        if v not in allowed:
            raise ValueError(f"Must be one of {allowed}")
        return v

    @validator("kyc_status")
    def valid_kyc(cls, v):
        if v not in [0, 1, 2]:
            raise ValueError("kyc_status: 0=pending, 1=verified, 2=rejected")
        return v


# ─── Response Schemas ─────────────────────────────────────────
class AlertInfo(BaseModel):
    """Step 6 — Alert & Notification Module"""
    user_alert   : str
    bank_alert   : str
    action_taken : str

class PredictionResponse(BaseModel):
    transaction_id    : str
    timestamp         : str
    # Step 5: Fraud Detection Module outputs
    fraud_probability : float
    risk_score        : int        # 0-100
    decision          : str        # ALLOW / REVIEW / BLOCK
    risk_level        : str        # LOW / MEDIUM / HIGH
    # Step 3: Feature Extraction — red flag reasons
    risk_factors      : List[str]
    # Step 6: Alert & Notification
    alert             : Optional[AlertInfo]
    # Metadata
    model_used        : str
    threshold_used    : float
    response_ms       : float

class ModelCompareResponse(BaseModel):
    transaction_id             : str
    logistic_regression_prob   : float
    random_forest_prob         : float
    gradient_boosting_prob     : float
    ensemble_avg_prob          : float
    final_decision             : str
    risk_level                 : str


# ─── Helpers ──────────────────────────────────────────────────
def preprocess_and_encode(data: TransactionRequest) -> tuple:
    """
    Step 2 — Data Preprocessing + Step 3 — Feature Extraction.
    Encode categoricals, normalize, return both raw and scaled vectors.
    """
    row = data.dict()

    # Encode categorical features (Step 2: preprocessing)
    for col in ["transaction_type", "transaction_city", "merchant_category"]:
        le  = encoders[col]
        val = row[col]
        row[col] = int(le.transform([val])[0]) if val in le.classes_ else 0

    # Build feature vector in correct order
    raw_vector = np.array([[row[f] for f in features]], dtype=float)

    # Normalize for Logistic Regression (Step 2: normalization)
    scaled_vector = scaler.transform(raw_vector)

    return raw_vector, scaled_vector


def get_risk_factors(data: TransactionRequest, prob: float) -> list:
    """
    Step 3 — Feature Extraction: identify behavioral red flags.
    Paper: unusual amount, odd hour, new device, location mismatch
    are red flag indicators.
    """
    flags = []
    if data.ip_risk_score > 0.5:
        flags.append(f"High IP risk score ({data.ip_risk_score:.2f})")
    if data.sim_change_flag:
        flags.append("SIM card recently changed — SIM swap risk")
    if data.pin_attempts >= 3:
        flags.append("Multiple PIN attempts — brute force risk")
    if data.is_new_device:
        flags.append("Transaction from unrecognized device")
    if data.location_mismatch:
        flags.append("Location does not match user profile")
    if data.txns_last_1hr >= 5:
        flags.append(f"High velocity: {data.txns_last_1hr} transactions in last hour")
    if data.is_odd_hour:
        flags.append(f"Unusual transaction hour: {data.hour_of_day}:00")
    if data.kyc_status == 2:
        flags.append("KYC status rejected")
    if data.is_new_payee:
        flags.append("Payment to new/unrecognized payee")
    if data.failed_txns_24hr >= 3:
        flags.append(f"Multiple failed transactions today ({data.failed_txns_24hr})")
    if abs(data.amount_deviation_from_avg) > 2.0:
        flags.append(f"Amount deviates significantly from spending history")
    if not flags and prob >= threshold:
        flags.append("Anomalous transaction pattern detected by ML model")
    return flags[:4]


def get_decision_and_risk(prob: float):
    """Step 5 — Fraud Detection Module: risk score threshold comparison."""
    if prob >= 0.60:
        return "BLOCK",  "HIGH",   min(100, int(prob * 100))
    elif prob >= threshold:
        return "REVIEW", "MEDIUM", int(prob * 100)
    else:
        return "ALLOW",  "LOW",    int(prob * 100)


def generate_alert(decision: str, risk_factors: list) -> Optional[AlertInfo]:
    """
    Step 6 — Alert & Notification Module.
    Paper: immediately notify user AND banking authority.
    """
    if decision == "ALLOW":
        return None
    if decision == "BLOCK":
        return AlertInfo(
            user_alert   = "🚨 Your transaction has been BLOCKED. "
                           "If not initiated by you, contact your bank immediately.",
            bank_alert   = "FRAUD ALERT: Transaction blocked. "
                           "Require immediate review — " + "; ".join(risk_factors[:2]),
            action_taken = "Transaction blocked. Additional authentication required."
        )
    return AlertInfo(
        user_alert   = "⚠️ Suspicious transaction detected. "
                       "Please verify this transaction in your UPI app.",
        bank_alert   = "REVIEW REQUIRED: Suspicious transaction flagged — "
                       + "; ".join(risk_factors[:2]),
        action_taken = "Transaction held for user verification."
    )


# ─── Routes ───────────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    return {
        "title"      : "UPI Fraud Detection API",
        "paper"      : "Real-Time UPI Fraud Detection Using Machine Learning (IJEDR 2026)",
        "pipeline"   : [
            "Step 1: Data Collection",
            "Step 2: Data Preprocessing",
            "Step 3: Feature Extraction",
            "Step 4: ML Model Prediction",
            "Step 5: Fraud Detection (Risk Score)",
            "Step 6: Alert & Notification",
            "Step 7: Reporting & Monitoring",
        ],
        "models"     : ["Logistic Regression", "Random Forest", "Gradient Boosting"],
        "best_model" : best_model_name,
        "docs"       : "/docs",
        "endpoints"  : {
            "predict"        : "POST /predict",
            "compare_models" : "POST /predict/compare",
            "batch"          : "POST /predict/batch",
            "stats"          : "GET /stats",
            "alerts"         : "GET /alerts",
            "health"         : "GET /health",
        }
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status"    : "ok",
        "best_model": best_model_name,
        "threshold" : round(float(threshold), 4),
        "uptime_s"  : round(time.time() - START_TIME, 1),
        "total_txns": tx_counter,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(txn: TransactionRequest):
    """
    Full research paper pipeline execution for a single transaction.
    Steps 1–7 all executed per request.
    """
    global tx_counter
    t0 = time.time()
    tx_counter += 1
    tx_id = f"TXN{tx_counter:08d}"

    # Step 2 + 3: Preprocess & extract features
    raw_vec, scaled_vec = preprocess_and_encode(txn)

    # Step 4: ML Model Prediction (best model)
    try:
        # Use scaled vector for Logistic Regression, raw for tree-based
        if "Logistic" in best_model_name:
            prob = float(best_model.predict_proba(scaled_vec)[0][1])
        else:
            prob = float(best_model.predict_proba(raw_vec)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # Step 5: Fraud Detection — risk score vs threshold
    decision, risk_level, risk_score = get_decision_and_risk(prob)

    # Step 3: Red flag reasons
    risk_factors = get_risk_factors(txn, prob)

    # Step 6: Alert & Notification
    alert = generate_alert(decision, risk_factors)

    # Step 7: Log transaction (Reporting & Monitoring)
    log_entry = {
        "transaction_id"   : tx_id,
        "timestamp"        : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fraud_probability": round(prob, 4),
        "decision"         : decision,
        "risk_level"       : risk_level,
        "model_used"       : best_model_name,
    }
    transaction_log.append(log_entry)
    if alert:
        alert_log.append({**log_entry, "alert": alert.dict()})

    return PredictionResponse(
        transaction_id    = tx_id,
        timestamp         = log_entry["timestamp"],
        fraud_probability = round(prob, 4),
        risk_score        = risk_score,
        decision          = decision,
        risk_level        = risk_level,
        risk_factors      = risk_factors if risk_factors else [],
        alert             = alert,
        model_used        = best_model_name,
        threshold_used    = round(float(threshold), 4),
        response_ms       = round((time.time() - t0) * 1000, 2),
    )


@app.post("/predict/compare", response_model=ModelCompareResponse, tags=["Prediction"])
def predict_compare(txn: TransactionRequest):
    """
    Paper comparison: Run all three models and compare results.
    Shows Logistic Regression vs Random Forest vs Gradient Boosting.
    """
    raw_vec, scaled_vec = preprocess_and_encode(txn)

    lr_prob = float(lr_model.predict_proba(scaled_vec)[0][1])
    rf_prob = float(rf_model.predict_proba(raw_vec)[0][1])
    gb_prob = float(gb_model.predict_proba(raw_vec)[0][1])
    ensemble_prob = (lr_prob + rf_prob + gb_prob) / 3

    decision, risk_level, _ = get_decision_and_risk(ensemble_prob)

    global tx_counter
    tx_counter += 1

    return ModelCompareResponse(
        transaction_id           = f"CMP{tx_counter:08d}",
        logistic_regression_prob = round(lr_prob, 4),
        random_forest_prob       = round(rf_prob, 4),
        gradient_boosting_prob   = round(gb_prob, 4),
        ensemble_avg_prob        = round(ensemble_prob, 4),
        final_decision           = decision,
        risk_level               = risk_level,
    )


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(transactions: list[TransactionRequest]):
    """Batch prediction — max 100 transactions."""
    if len(transactions) > 100:
        raise HTTPException(status_code=400,
                            detail="Batch size limit is 100 transactions.")
    return [predict(txn) for txn in transactions]


@app.get("/alerts", tags=["Step 6 — Alerts"])
def get_alerts(limit: int = 20):
    """Step 6: View recent fraud alerts sent to users and bank."""
    return {
        "total_alerts": len(alert_log),
        "recent_alerts": alert_log[-limit:][::-1],
    }


@app.get("/stats", tags=["Step 7 — Monitoring"])
def stats():
    """Step 7: Reporting & Monitoring — transaction statistics."""
    if not transaction_log:
        return {"message": "No transactions processed yet."}

    decisions = [t["decision"] for t in transaction_log]
    total     = len(decisions)

    return {
        "step"              : "7 — Reporting & Monitoring Module",
        "total_transactions": total,
        "allow_count"       : decisions.count("ALLOW"),
        "review_count"      : decisions.count("REVIEW"),
        "block_count"       : decisions.count("BLOCK"),
        "fraud_rate_pct"    : round((total - decisions.count("ALLOW")) / total * 100, 2)
                               if total > 0 else 0,
        "alert_count"       : len(alert_log),
        "best_model"        : best_model_name,
        "threshold"         : round(float(threshold), 4),
        "features_used"     : len(features),
        "uptime_seconds"    : round(time.time() - START_TIME, 1),
        "note"              : "Transaction logs used for periodic model retraining (paper Step 7)",
    }


@app.get("/pipeline", tags=["Info"])
def pipeline_info():
    """Research paper pipeline modules explanation."""
    return {
        "paper": "Real-Time UPI Fraud Detection Using Machine Learning — IJEDR 2026",
        "modules": {
            "Step 1 — Data Collection"    : "Collects transaction amount, time, device ID, location, user history",
            "Step 2 — Data Preprocessing" : "Removes duplicates, encodes categoricals, normalizes features",
            "Step 3 — Feature Extraction" : "Extracts behavioral patterns, identifies red flag indicators",
            "Step 4 — ML Model Training"  : "Logistic Regression + Random Forest + Gradient Boosting",
            "Step 5 — Fraud Detection"    : f"Risk score generated, compared against threshold ({threshold:.4f})",
            "Step 6 — Alert & Notification": "Instantly notifies user AND banking authority on suspicious activity",
            "Step 7 — Reporting & Monitoring": "Logs all transactions, generates fraud trend reports, enables model retraining",
        },
        "models": {
            "Logistic Regression" : "Best for basic statistical classification",
            "Random Forest"       : "Most stable — combines multiple behavioural features",
            "Gradient Boosting"   : "Detects more delicate fraudulent signals",
            "Best (auto-selected)": best_model_name,
        }
    }