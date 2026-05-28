# UPI Fraud Detection System

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

**A Machine Learning based system to detect fraudulent UPI transactions in real-time.**

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🌟 Overview

With the rapid growth of **UPI (Unified Payments Interface)** in India, fraud cases have also increased significantly. This project aims to build an intelligent system that can **detect fraudulent UPI transactions** using Machine Learning.

The model analyzes various transaction features like amount, time, merchant type, device info, transaction pattern, etc., to classify a transaction as **Fraudulent** or **Legitimate**.

---

## ✨ Features

- Real-time UPI fraud prediction
- Interactive web interface using **Streamlit**
- Multiple ML models comparison (Random Forest, XGBoost, LightGBM, etc.)
- Feature importance analysis
- Transaction history & analytics dashboard
- High accuracy with class imbalance handling
- Model explainability using SHAP

---

## 📊 Dataset

- **Source**: Synthetic + Real-world inspired UPI transaction data
- **Total Records**: ~1,00,000+
- **Class Distribution**: Highly imbalanced (Fraud vs Legitimate)
- **Key Features**: Transaction Amount, Time, Location, Device Type, Merchant Category, Previous Transaction History, etc.

*(Dataset is available in the `/data` folder)*

---

## 🛠 Technologies Used

- **Language**: Python 3.9+
- **ML Libraries**: Scikit-learn, XGBoost, LightGBM, Imbalanced-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Web App**: Streamlit
- **Model Explainability**: SHAP
- **Version Control**: Git & GitHub

---

## 📈 Model Performance

| Model                | Accuracy | Precision | Recall  | F1-Score | AUC-ROC |
|----------------------|----------|-----------|---------|----------|---------|
| XGBoost              | 98.7%    | 0.96      | 0.89    | 0.92     | 0.98    |
| Random Forest        | 97.9%    | 0.94      | 0.85    | 0.89     | 0.97    |
| LightGBM             | 98.4%    | 0.95      | 0.88    | 0.91     | 0.98    |

**Best Model**: XGBoost (Used in deployment)

---

## 📁 Project Structure
UPI_Fraud_Detection/ /n
├── data/ /n
├── notebooks/ /n
├── models/  /n
├── src/  /n
│   ├── preprocessing.py  /n
│   ├── feature_engineering.py  /n
│   ├── model_training.py   /n
│   └── utils.py  /n
├── streamlit_app.py /n
├── requirements.txt /n
├── README.md /n
└── .gitignore
