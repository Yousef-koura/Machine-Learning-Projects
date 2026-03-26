# 🤖 Machine Learning — Complete Reference Guide

> A personal reference covering the full ML pipeline from EDA to deployment.

---

## 📋 Table of Contents
1. [EDA — Exploratory Data Analysis](#eda)
2. [Preprocessing](#preprocessing)
3. [Models](#models)
4. [Optimization Techniques](#optimization)
5. [Evaluation Metrics](#evaluation)

---

<a name="eda"></a>
## 📊 1. EDA — Exploratory Data Analysis

### Essential EDA Commands
```python
df.head()           # first 5 rows
df.shape            # (rows, columns)
df.describe()       # statistics: mean, std, min, max
df.isnull().sum()   # count missing values
df.dtypes           # column data types
df['col'].value_counts()  # class distribution
```

### Types of Plots

| Plot | Code | Shows | Use When |
|------|------|-------|----------|
| **Histogram** | `sns.histplot(df['col'], kde=True)` | Distribution shape, skewness | Check if data is normal or skewed |
| **Boxplot** | `sns.boxplot(x=df['col'])` | Median, spread, outliers | Detect outliers |
| **Violin Plot** | `sns.violinplot(x=df['col'])` | Distribution + boxplot combined | More detail than boxplot |
| **Scatter Plot** | `plt.scatter(y_actual, y_pred)` | Relationship between 2 variables | Actual vs Predicted (regression) |
| **Heatmap** | `sns.heatmap(df.corr(), annot=True)` | Correlation between ALL features | Feature selection |
| **Pair Plot** | `sns.pairplot(df)` | All feature relationships at once | Full EDA overview |
| **Bar Chart** | `sns.countplot(x=df['col'])` | Class frequency | Check class imbalance |
| **Line Chart** | `plt.plot(x, y)` | Trend over time | Time series data |

### Reading Correlation (Heatmap)
```
value close to  1.0  →  strong positive correlation  📈 (both increase together)
value close to -1.0  →  strong negative correlation  📉 (one increases, other decreases)
value close to  0.0  →  no correlation               ➡️
```

### Reading Skewness (Histogram)
```
mean ≈ median  →  Normal distribution  ✅
mean > median  →  Right skewed (positive) — few very high values
mean < median  →  Left skewed (negative)  — few very low values
```

---

<a name="preprocessing"></a>
## 🔧 2. Preprocessing

### 2.1 — Handling Missing Values

```python
df.isnull().sum()                          # detect missing values
df.isnull().sum() / len(df) * 100          # percentage missing
```

| Method | Code | Best For |
|--------|------|----------|
| **Drop rows** | `df.dropna()` | Missing < 5% of data |
| **Drop columns** | `df.dropna(axis=1)` | Column has too many missing values |
| **Fill with Mean** | `df['col'].fillna(df['col'].mean())` | Numerical, no outliers |
| **Fill with Median** | `df['col'].fillna(df['col'].median())` | Numerical with outliers ✅ |
| **Fill with Mode** | `df['col'].fillna(df['col'].mode()[0])` | Categorical columns |
| **Fill with Value** | `df.fillna(0)` or `df.fillna('')` | Specific known value |
| **Forward Fill** | `df.fillna(method='ffill')` | Time series data |
| **Backward Fill** | `df.fillna(method='bfill')` | Time series data |
| **KNN Imputer** | `KNNImputer(n_neighbors=5)` | Complex datasets, most accurate |
| **Interpolation** | `df['col'].interpolate()` | Sequential/time series data |

> ⚠️ **Rule:** Always handle missing values BEFORE encoding or scaling.

---

### 2.2 — Handling Outliers

```python
# detect outliers
sns.boxplot(x=df['col'])           # visual detection
df.describe()                       # check min/max vs mean
```

| Method | Code | Best For |
|--------|------|----------|
| **IQR Method** | Remove rows where value < Q1-1.5×IQR or > Q3+1.5×IQR | Most common ✅ |
| **Z-Score** | Remove rows where \|z-score\| > 3 | Normally distributed data |
| **Cap/Clip** | `df['col'].clip(lower, upper)` | Keep rows but limit extreme values |
| **Log Transform** | `np.log1p(df['col'])` | Reduce impact of outliers |
| **RobustScaler** | `RobustScaler()` | Scale without removing outliers |

```python
# IQR Method
# first count the outliers for age
Q1 = data['age'].quantile(0.25)
Q3 = data['age'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR

age_outliers = data[(data['age'] < lower_bound) | (data['age'] > upper_bound)]

# Z-Score Method
from scipy import stats
df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
```

---

### 2.3 — Encoding Categorical Data

> **Rule:** ML models only understand numbers — text must be converted.

| Method | Code | Best For |
|--------|------|----------|
| **map()** | `df['col'].map({'Yes':1, 'No':0})` | Binary, full control over mapping |
| **LabelEncoder** | `LabelEncoder().fit_transform(df['col'])` | Target column (y) |
| **One Hot Encoding** | `pd.get_dummies(df, columns=['col'])` | No ranking categories (city, color) |
| **Ordinal Encoding** | `OrdinalEncoder(categories=[['Low','Med','High']])` | Ranked categories |
| **replace()** | `df.replace({'col': {'A':1, 'B':2}})` | Multiple columns at once |

```python
# Binary encoding with map()
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0}).astype(int)

# One Hot Encoding
df = pd.get_dummies(df, columns=['Property_Area'])

# Ordinal Encoding
df['Size'] = df['Size'].map({'Small': 0, 'Medium': 1, 'Large': 2}).astype(int)
```

> ⚠️ **Always add `.astype(int)`** after `map()` to avoid string type issues with models.

---

### 2.4 — Standardization / Feature Scaling

> **Rule:** Models like SVM, Logistic Regression, KNN are sensitive to feature scale.

| Method | Formula | Range | Sensitive to Outliers | Best For |
|--------|---------|-------|-----------------------|----------|
| **StandardScaler** | (x - mean) / std | ~-3 to +3 | ❌ Yes | SVM, Logistic Regression, Neural Networks |
| **MinMaxScaler** | (x - min) / (max - min) | 0 to 1 | ❌ Yes | Neural Networks, image data |
| **RobustScaler** | (x - median) / IQR | varies | ✅ No | Data with outliers |
| **Normalizer** | scales each row to length 1 | 0 to 1 | ✅ No | NLP, text data |
| **Log Transform** | log(x + 1) | varies | ✅ No | Highly skewed data |

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)   # learn scale from train ✅
x_test  = scaler.transform(x_test)        # apply same scale ✅
# never use fit_transform on test data ❌
```

---

### 2.5 — Handling Class Imbalance

```python
df['target'].value_counts()   # check class balance
```

| Method | Code | Description |
|--------|------|-------------|
| **class_weight='balanced'** | `SVC(class_weight='balanced')` | Penalize majority class more ✅ easy |
| **Oversampling (SMOTE)** | `SMOTE().fit_resample(x, y)` | Generate synthetic minority samples |
| **Undersampling** | `RandomUnderSampler()` | Remove majority class samples |
| **stratify=y** | `train_test_split(..., stratify=y)` | Keep class ratio in split ✅ always use |

```python
# easiest fix
classifier = SVC(kernel='linear', class_weight='balanced')

# SMOTE
from imblearn.over_sampling import SMOTE
x_res, y_res = SMOTE().fit_resample(x_train, y_train)
```

---

### 2.6 — Data Splitting

| Split | Code | Best For |
|-------|------|----------|
| **Train / Test** | `train_test_split(x, y, test_size=0.2)` | Quick baseline |
| **Train / Val / Test** | Two-step split | Hyperparameter tuning |
| **Cross Validation** | `cross_val_score(model, x, y, cv=5)` | Small datasets ✅ |
| **Stratified Split** | `train_test_split(..., stratify=y)` | Imbalanced classes |

```python
# Train / Validation / Test
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=1)
x_val, x_test, y_val, y_test     = train_test_split(x_temp, y_temp, test_size=0.5, random_state=1)
# result: 70% train, 15% val, 15% test
```

---

<a name="models"></a>
## 🤖 3. Models

### 3.1 — Regression Models (predict continuous values)

| Model | Import | Best For |
|-------|--------|----------|
| **Linear Regression** | `from sklearn.linear_model import LinearRegression` | Simple linear relationships |
| **Ridge Regression** | `from sklearn.linear_model import Ridge` | Linear + L2 regularization |
| **Lasso Regression** | `from sklearn.linear_model import Lasso` | Linear + L1 regularization (feature selection) |
| **XGBoost Regressor** | `from xgboost import XGBRegressor` | Complex tabular data ✅ |
| **Random Forest Regressor** | `from sklearn.ensemble import RandomForestRegressor` | Non-linear relationships |
| **SVR** | `from sklearn.svm import SVR` | Small to medium datasets |

---

### 3.2 — Classification Models (predict categories)

| Model | Import | Best For |
|-------|--------|----------|
| **Logistic Regression** | `from sklearn.linear_model import LogisticRegression` | Binary classification, baseline ✅ |
| **SVM** | `from sklearn import svm` | Small-medium datasets, high accuracy |
| **Decision Tree** | `from sklearn.tree import DecisionTreeClassifier` | Interpretable model |
| **Random Forest** | `from sklearn.ensemble import RandomForestClassifier` | General purpose ✅ |
| **XGBoost** | `from xgboost import XGBClassifier` | Competitions, best accuracy 🏆 |
| **KNN** | `from sklearn.neighbors import KNeighborsClassifier` | Simple, small datasets |
| **Naive Bayes** | `from sklearn.naive_bayes import GaussianNB` | NLP, text classification |

### How They Work (Brief)

```
Logistic Regression → Z = w*x + b → sigmoid(Z) → probability → 0 or 1
SVM                 → finds best hyperplane that maximizes margin between classes
Decision Tree       → series of if/else questions on features
Random Forest       → many trees vote in parallel → final answer
XGBoost             → trees built sequentially, each fixes errors of previous
KNN                 → finds K nearest neighbors → majority vote
```

---

<a name="optimization"></a>
## ⚙️ 4. Optimization Techniques

### 4.1 — Hyperparameter Tuning

| Method | Code | Description |
|--------|------|-------------|
| **GridSearchCV** | `GridSearchCV(model, param_grid, cv=5)` | Try ALL combinations ✅ thorough |
| **RandomizedSearchCV** | `RandomizedSearchCV(model, param_dist, n_iter=10)` | Try RANDOM combinations, faster |
| **Cross Validation** | `cross_val_score(model, x, y, cv=5)` | Evaluate model reliability |

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

grid = GridSearchCV(svm.SVC(), param_grid, cv=5)
grid.fit(x_train, y_train)

print(f"Best params : {grid.best_params_}")
print(f"Best score  : {grid.best_score_:.2f}")
```

---

### 4.2 — Regularization

| Method | Description | Used In |
|--------|-------------|---------|
| **L1 (Lasso)** | Shrinks some weights to zero → feature selection | Lasso, LogisticRegression |
| **L2 (Ridge)** | Shrinks all weights evenly → prevents overfitting | Ridge, SVM (C parameter) |
| **Dropout** | Randomly disables neurons during training | Neural Networks |
| **Early Stopping** | Stop training when validation loss stops improving | Neural Networks, XGBoost |

---

### 4.3 — Gradient Descent Variants

| Type | Description |
|------|-------------|
| **Batch GD** | Uses ALL data per update — slow but stable |
| **Stochastic GD** | Uses ONE sample per update — fast but noisy |
| **Mini-Batch GD** | Uses small batches — best of both ✅ most common |

---

### 4.4 — Overfitting vs Underfitting

```
Training accuracy >> Testing accuracy  →  Overfitting  ❌  model memorized data
Training accuracy ≈  Testing accuracy  →  Healthy      ✅
Both accuracies low                    →  Underfitting ❌  model didn't learn enough
```

| Problem | Fix |
|---------|-----|
| Overfitting | More data, regularization, dropout, reduce model complexity |
| Underfitting | More features, increase model complexity, more training |

---

<a name="evaluation"></a>
## 📈 5. Evaluation Metrics

### 5.1 — Classification Metrics

| Metric | Code | What it Measures |
|--------|------|-----------------|
| **Accuracy** | `accuracy_score(y_test, y_pred)` | % of correct predictions |
| **Precision** | `precision_score(y_test, y_pred)` | Of predicted positives, how many are actually positive |
| **Recall** | `recall_score(y_test, y_pred)` | Of actual positives, how many did we catch |
| **F1 Score** | `f1_score(y_test, y_pred)` | Balance between Precision and Recall ✅ |
| **ROC-AUC** | `roc_auc_score(y_test, y_prob)` | Model's ability to separate classes |
| **Confusion Matrix** | `confusion_matrix(y_test, y_pred)` | TP, TN, FP, FN breakdown |

```
Confusion Matrix:
                 Predicted 0    Predicted 1
Actual 0    →   TN (correct)   FP (wrong)
Actual 1    →   FN (wrong)     TP (correct)

Accuracy  = (TP + TN) / Total
Precision = TP / (TP + FP)   → how precise are positive predictions?
Recall    = TP / (TP + FN)   → how many positives did we catch?
F1        = 2 × (Precision × Recall) / (Precision + Recall)
```

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# full report
print(classification_report(y_test, y_pred))

# confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['No', 'Yes'],
            yticklabels=['No', 'Yes'])
```

---

### 5.2 — Regression Metrics

| Metric | Code | What it Measures |
|--------|------|-----------------|
| **R² Score** | `r2_score(y_test, y_pred)` | How well model explains data (1.0 = perfect) |
| **MAE** | `mean_absolute_error(y_test, y_pred)` | Average prediction error |
| **MSE** | `mean_squared_error(y_test, y_pred)` | Penalizes large errors more |
| **RMSE** | `np.sqrt(mean_squared_error(...))` | Same as MSE but in original units |

```
R² = 1.00  →  perfect model 🎯
R² = 0.85+ →  good model ✅
R² = 0.50  →  weak model
R² = 0.00  →  model learned nothing ❌

MAE / RMSE close to 0 → accurate predictions ✅
```

---

### 5.3 — ROC Curve

```python
from sklearn.metrics import roc_curve, auc

y_prob = model.decision_function(x_test)   # SVM
# y_prob = model.predict_proba(x_test)[:,1]  # other models

fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
plt.plot([0,1], [0,1], 'r--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

```
AUC = 1.0  →  perfect model 🎯
AUC = 0.8+ →  good model ✅
AUC = 0.5  →  random guessing ❌
```

---

### 5.4 — Cross Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, x, y, cv=5, scoring='accuracy')
print(f"CV Scores  : {scores}")
print(f"Mean Score : {scores.mean():.2f} ± {scores.std():.2f}")
```

```
high mean + low std  →  stable reliable model ✅
high mean + high std →  inconsistent model ⚠️
```

---

## 🗺️ Full ML Pipeline Summary

```
Raw Data
   ↓
1. EDA          → understand data, visualize, find patterns
   ↓
2. Preprocessing → missing values, outliers, encoding, scaling, balance
   ↓
3. Split Data   → train / validation / test
   ↓
4. Train Model  → fit on training data
   ↓
5. Evaluate     → accuracy, F1, ROC, R², MAE
   ↓
6. Optimize     → GridSearchCV, cross validation, regularization
   ↓
7. Final Test   → evaluate on test set (only once!) ✅
   ↓
8. Deploy       → save model → Streamlit / Flask
```

---

> 📌 **Personal Note:** Projects completed — Rock vs Mine (Logistic Regression), Diabetes (SVM), Housing Price (XGBoost), Fake News (NLP + Logistic Regression), Loan Prediction (SVM)
