# Multi-Label Healthcare Disease Prediction

A machine learning project designed to predict the likelihood of 10 different medical conditions simultaneously using patient demographics and clinical metrics. This project demonstrates handling extreme class imbalance in a multi-label classification setting.

## 🏥 Project Overview
In medical diagnostic modeling, datasets are often highly imbalanced—the majority of patients are healthy for a specific condition. A standard model often defaults to predicting "Healthy" for everyone to achieve high accuracy, but it fails to identify actual sick patients (zero recall). 

This project explores the transition from a standard **Random Forest** baseline to a highly optimized **XGBoost** pipeline that uses **Threshold Tuning** to prioritize patient safety (Recall).

## 📊 Dataset
The data used in this project is the **Healthcare Disease Prediction Dataset** from Kaggle.

- **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/algozee/healthcare-disease-prediction-dataset/data)
- **Size:** 1,000 patient records.
- **Features:** Age, Gender, BMI, Blood Pressure, Cholesterol, Glucose, Smoking status, Alcohol Consumption, Exercise, and Family History.
- **Targets:** 10 diseases (Heart Disease, Diabetes, Stroke, Kidney Disease, Cancer, Alzheimer's, COPD, Liver Disease, Parkinson's, and Tuberculosis).

## 🚀 Technical Highlights
- **Multi-Label Strategy:** Utilizes `MultiOutputClassifier` to handle 10 binary targets at once.
- **Model Evolution:** - **Baseline:** Random Forest with `class_weight='balanced'`. 
  - **Optimized:** XGBoost (Extreme Gradient Boosting) for its sequential learning capability.
- **Threshold Tuning:** Shifted the classification threshold from the default **0.5** down to **0.4**. This resulted in a significant boost in **Recall**, ensuring the model actually "detects" disease states instead of ignoring them.

## 📈 Final Results (Threshold 0.4)
The final optimized XGBoost model achieved:
- **Heart Disease:** 95% Recall
- **Liver Disease:** 72% Recall
- **Alzheimer's Disease:** 64% Recall
- **Overall:** Successfully pulled 9 out of 10 diseases out of the "Zero Recall" zone.

## ⚙️ Installation & Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
