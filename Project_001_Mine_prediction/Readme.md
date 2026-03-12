# 🪨 Rock vs Mine Detection — Sonar Data Classification

## 📌 Project Overview
A machine learning project that classifies underwater objects as **rocks or mines** using sonar signal data. The model is trained on the UCI Sonar dataset and uses logistic regression to distinguish between the two classes based on frequency response patterns.

## 🔄 Workflow

<p align="center">
  <img src="workflow.png" alt="Project Workflow" width="600"/>
</p>

| Step | Description |
|------|-------------|
| 📥 Data Collection | Sonar dataset containing 208 samples and 60 frequency features |
| 🧹 Data Preprocessing | Handling missing values, feature scaling, and encoding labels |
| ✂️ Data Splitting | Dividing data into training and testing sets (80/20 split) |
| 🤖 Model Training | Logistic Regression model trained on sonar features |
| 📊 Evaluation | Measuring accuracy, precision, recall, and confusion matrix |

## 🛠️ Tech Stack
![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data-green)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-lightblue)

## 📁 Project Structure
```
├── data/
│   └── sonar.csv
├── notebook/
│   └── rock_vs_mine.ipynb
├── workflow.png
└── README.md
```

## 📈 Results
| Metric | Score |
|--------|-------|
| Training Accuracy | XX% |
| Testing Accuracy  | XX% |