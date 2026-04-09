# 🛡️ UPI Fraud Detection System

Real-time UPI transaction fraud detection using Machine Learning.

## 📁 Project Structure

```
UPI_Fraud_Detection/
├── data/
│   └── upi_fraud_dataset_105k.csv
├── models/                        ← auto-created after training
│   ├── hist_gb_model.pkl
│   ├── threshold.pkl
│   ├── label_encoders.pkl
│   └── feature_names.pkl
├── plots/                         ← auto-created after training
│   └── confusion_and_pr_curve.png
├── api/
│   └── main.py                    ← FastAPI backend
├── dashboard/
│   └── app.py                     ← Streamlit dashboard
├── notebooks/
│   └── data.ipynb                 ← EDA + model training
├── phase1_train.py                ← SMOTE + Optuna + save model
├── requirements.txt
└── README.md
```

## 🚀 Setup

```bash
pip install -r requirements.txt
```

## 📋 Run Order

### Step 1 — Train the model
```bash
python phase1_train.py
```
This will:
- Load the dataset
- Apply SMOTE to fix class imbalance
- Run Optuna (60 trials) to find best hyperparameters
- Tune the decision threshold via Precision-Recall curve
- Save model artifacts to `models/`
- Save confusion matrix + PR curve to `plots/`

### Step 2 — Start FastAPI backend
```bash
uvicorn api.main:app --reload --port 8000
```
API docs available at: http://localhost:8000/docs

### Step 3 — Launch Streamlit dashboard
```bash
streamlit run dashboard/app.py
```
Dashboard at: http://localhost:8501

## 🔌 API Usage

### POST /predict
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_type": "P2P",
    "transaction_city": "Delhi",
    "hour_of_day": 3,
    "is_odd_hour": 1,
    "is_new_device": 1,
    "is_new_payee": 1,
    "location_mismatch": 1,
    "txns_last_1hr": 8,
    "pin_attempts": 3,
    "ip_risk_score": 0.87,
    "sim_change_flag": 1,
    "kyc_status": 2,
    "merchant_category": "peer"
  }'
```

### Response
```json
{
  "transaction_id": "TXN00000001",
  "timestamp": "2024-10-15 03:22:11",
  "fraud_probability": 0.9234,
  "decision": "BLOCK",
  "risk_level": "HIGH",
  "risk_score": 92,
  "reasons": [
    "High IP risk score (0.87)",
    "SIM card recently changed (SIM swap risk)",
    "Multiple PIN attempts (brute force risk)",
    "Transaction from unrecognized device"
  ],
  "response_ms": 4.2
}
```

## 🤖 Model

| Property | Value |
|---|---|
| Algorithm | HistGradientBoostingClassifier |
| Training data | 105,000 transactions |
| Class balancing | SMOTE (sampling_strategy=0.25) |
| Tuning | Optuna (60 trials, 3-fold CV) |
| Metric optimized | ROC-AUC |
| Threshold | Tuned via F1-optimal PR curve |

## 📊 Risk Levels

| Decision | Probability | Action |
|---|---|---|
| ALLOW | < threshold | Transaction proceeds |
| REVIEW | threshold – 0.60 | Flagged for review |
| BLOCK | ≥ 0.60 | Transaction halted |
