# UPI Fraud Detection System

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

**A Machine Learning based real-time UPI Fraud Detection System**

---

## 📋 Table of Contents
- [Overview](#overview)
- [Demo Screenshots](#demo-screenshots)
- [Features](#features)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## 🌟 Overview

With the massive growth of **UPI transactions** in India, fraud cases have also increased significantly. This project aims to detect **fraudulent UPI transactions** in real-time using Machine Learning.

The system analyzes various transaction features and predicts whether a transaction is **Fraudulent** or **Legitimate**.

---

## 📸 Demo Screenshots

*(Add your screenshots here)*

![Streamlit Dashboard](images/dashboard.png)  
![Fraud Prediction Result](images/prediction.png)  
![Model Performance](images/performance.png)

---

## ✨ Features

- Real-time UPI fraud prediction
- Interactive Streamlit Web Application
- Multiple ML Models (XGBoost, Random Forest, LightGBM)
- Feature Importance Visualization
- Transaction Analytics Dashboard
- Download Prediction Report

---

## 📊 Dataset

- **Source**: Synthetic + Real-world inspired UPI transaction data
- **Records**: ~1,00,000+
- **Target Variable**: Fraud (1) / Legitimate (0)

---

## 📈 Model Performance

| Model                | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|----------------------|----------|-----------|--------|----------|---------|
| **XGBoost**          | **98.7%**| **0.96**  | 0.89   | **0.92** | **0.98** |
| LightGBM             | 98.4%    | 0.95      | 0.88   | 0.91     | 0.98    |
| Random Forest        | 97.9%    | 0.94      | 0.85   | 0.89     | 0.97    |

---

## 🛠 Technologies Used

- **Language**: Python
- **ML Frameworks**: Scikit-learn, XGBoost, LightGBM
- **Frontend**: Streamlit
- **Data Handling**: Pandas, NumPy
- **Visualization**: Plotly, Seaborn, Matplotlib

---

## 📁 Project Structure


UPI_Fraud_Detection/
├── data/
├── notebooks/
├── models/
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── utils.py
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .gitignore


# Clone the repository
git clone https://github.com/Prakhar-garg12/UPI_Fraud_Detection.git
cd UPI_Fraud_Detection

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py
