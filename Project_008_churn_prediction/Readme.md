# 📊 Customer Churn Prediction — Classification Using Multiple ML Models

## 📌 Project Overview
A machine learning project that predicts whether a **telecom customer will churn (leave)** based on their account information and usage patterns. The project compares 4 models with hyperparameter tuning via **RandomizedSearchCV** and **Stratified K-Fold Cross Validation**, with the best model saved using **Pickle** for deployment.

---

## 🔄 Workflow

| Step | Description |
|------|-------------|
| 📥 Data Collection | Telco Customer Churn dataset — 7043 samples and 21 features |
| 🧹 Understand Data | Checked shape, dtypes, missing values, unique values, class balance |
| 🔧 Preprocessing | Dropped customerID, fixed TotalCharges dtype, encoded categorical columns |
| 📊 EDA | Distribution plots, boxplots, correlation heatmap |
| 🔢 Encoding | Label Encoding per column with saved encoders dictionary |
| ✂️ Data Splitting | 80/20 split with stratify=y to maintain class ratio |
| ⚖️ Scaling | StandardScaler on numerical columns only (tenure, MonthlyCharges, TotalCharges) |
| ⚖️ Handle Imbalance | class_weight='balanced' for tree models, scale_pos_weight for XGBoost |
| 🤖 Model Training | 4 models compared with RandomizedSearchCV + Stratified K-Fold |
| 📊 Evaluation | Accuracy, classification report, confusion matrix, ROC curve |
| 💾 Save Model | Saved best model + scaler + encoders using Pickle |
| 🔮 Prediction | Predicts churn or no churn for a new customer input |

---

## 🛠️ Tech Stack
![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-red)
![Pandas](https://img.shields.io/badge/Pandas-Data-green)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-lightblue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-purple)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-teal)

---

### Why Stratified K-Fold?
```
Regular KFold    →  random split → some folds may have 90% class 0 ❌
Stratified KFold →  every fold maintains same 73/27 ratio ✅
```

### Why F1 Scoring?
```
Accuracy → misleading for imbalanced data ❌
F1       → balances precision and recall ✅
```
---

## 📁 Project Structure
```
├── churn_data.csv          (dataset)
├── model.ipynb             (model code)
├── churn_model.pkl         (saved best model)
├── workflow.png
└── README.md               (project description)
```
---

## 📈 Results

| Metric | Logistic Regression | Decision Tree | Random Forest | XGBoost |
|--------|--------------------:|-------------:|--------------:|--------:|
| Test Accuracy | 74% | 76% | 77% | 74% |
| AUC Score | 0.84 | 0.82 | 0.84 | 0.85 |