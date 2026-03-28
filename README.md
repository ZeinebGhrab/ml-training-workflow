# 📁 Data Science & Machine Learning Projects

A collection of four data science lab projects covering unsupervised clustering, regression-based house price prediction, and binary classification for customer churn prediction.

---

## 📂 Project Structure

```
├── Crimes Clustering/
│   ├── crimes.csv
│   ├── cluster_assignments.csv
│   ├── cluster_centroids.csv
│   ├── cluster_statistics.csv
│   ├─ cluster_summary.csv
│   └──crimes_clustering_analysis.ipynb
├── TP1/
│   ├──Project_Description.pdf
│   ├──data_description.txt
│   ├──description_notebook_eda.pdf
│   ├──eda.ipynb
│   ├──eda1.ipynb
│   ├──sample_submission.csv
│   └──train.csv
├── TP2/
│   ├── Remarques.txt
│   └── eda2_copy.ipyn
└── TP3/
    ├──Churn_Modelling.csv
    ├──churn_analysis (2).ipynb
    ├──churn_analysis (3).ipynb
    ├──churn_analysis_(3).ipynb
    └── guide_projet_churn.md
```

---

## 🔵 Crimes Clustering — Unsupervised Learning (K-Means)

### Overview

K-Means clustering applied to US state-level crime statistics to group states by their crime profiles across 7 crime categories.

### Dataset

- **Source:** US crime statistics per state
- **Features:** Murder, Rape, Robbery, Assault, Burglary, Larceny, Auto Theft
- **Scope:** 50 US states

### Clusters

| Cluster | # States | Profile |
|---------|----------|---------|
| 0 | 11 | **High crime** — high rates across all categories; includes CA, NY, NV, FL |
| 1 | 16 | **Low crime** — lowest rates overall; rural/midwest states (IA, ND, VT…) |
| 2 | 17 | **Medium-high crime** — elevated assault and murder; southern states (AL, TX, LA…) |
| 3 | 6  | **Medium crime with high auto theft** — CT, MA, NJ, HI, DE, RI |

### Key Files

| File | Description |
|------|-------------|
| `crimes.csv` | Raw crime data (semicolon-separated) |
| `cluster_assignments.csv` | State → cluster mapping |
| `cluster_centroids.csv` | Mean values per cluster per feature |
| `cluster_statistics.csv` | Full descriptive statistics per cluster |
| `cluster_summary.csv` | Cluster sizes and state lists |

---

## 🟡 TP1 — House Price Prediction (Regression)

### Overview

Supervised regression task predicting residential house sale prices using the Ames Housing Dataset.

### Dataset

- **Target variable:** `SalePrice`
- **Size:** ~1,460 properties
- **Features:** 79 explanatory variables covering structural, geographic, and quality attributes

### Feature Categories

- **Structural:** `MSSubClass`, `BldgType`, `HouseStyle`, `YearBuilt`, `GrLivArea`, `TotalBsmtSF`
- **Quality ratings:** `OverallQual`, `OverallCond`, `ExterQual`, `KitchenQual`
- **Lot & location:** `LotArea`, `LotFrontage`, `Neighborhood`, `MSZoning`
- **Amenities:** `GarageCars`, `Fireplaces`, `PoolArea`, `WoodDeckSF`

### Notes

See `data_description.txt` for the full codebook of all 79 variables and their possible values.

---

## 🟢 TP2 — House Price Prediction (EDA & Preprocessing Notes)

### Overview

Supplementary notes on exploratory data analysis and preprocessing decisions for the regression task.

### Key Concepts Covered

**Boxplot Interpretation**
- Median position → symmetry or skewness
- IQR size → data spread
- Whisker length → distribution tails
- Outliers → extreme values

**Target Variable**
- `SalePrice` is right-skewed
- **Fix:** log transformation to normalize distribution, stabilize variance, and improve model convergence

**Standardization**
- Z-score normalization: `z = (x - mean) / std`
- Applied before distance-based or gradient-based models

**Models Used**
- Linear Regression (no regularization, sensitive to overfitting)
- Ridge Regression (L2 regularization — reduces coefficient magnitude and variance)

---

## 🔴 TP3 — Customer Churn Prediction (Binary Classification)

### Overview

End-to-end machine learning pipeline to predict whether a bank customer will churn (leave), enabling targeted retention strategies.

### Dataset

- **Source:** `Churn_Modelling.csv`
- **Size:** 10,000 customers, 14 columns
- **Target:** `Exited` (0 = retained, 1 = churned)
- **Class imbalance:** ~80% retained / ~20% churned

### Pipeline

```
EDA → Preprocessing → Modeling → Fine-tuning → Feature Selection → Recommendations
```

#### Phase 1 — EDA
- Distribution analysis, outlier detection (boxplots)
- Correlation heatmap, pairplots
- Churn rate by geography, gender, product count

#### Phase 2 — Preprocessing
- Drop non-predictive columns: `RowNumber`, `CustomerId`, `Surname`
- Encode `Gender` (Label Encoding), `Geography` (One-Hot Encoding, drop_first)
- Train/test split: 80/20, stratified, `random_state=42`
- Normalize with `StandardScaler` (fit on train only)

#### Phase 3 — Models Compared
| Model | Notes |
|-------|-------|
| K-Nearest Neighbors | Distance-based, requires normalization |
| Logistic Regression | Linear baseline, interpretable |
| Decision Tree | Visualizable rules, prone to overfitting |
| Random Forest | Ensemble, robust, feature importance |
| XGBoost | Gradient boosting, state-of-the-art |

#### Phase 4 — Hyperparameter Tuning
- Method: `GridSearchCV` with 5-fold cross-validation
- Tuned: `C`, `n_estimators`, `max_depth`, `learning_rate`, `subsample`, etc.

#### Phase 5 — Feature Selection
- Method: `SelectKBest` with `f_classif` (ANOVA F-score)
- K values tested: 5, 7, 10
- Top features: `Age`, `NumOfProducts`, `IsActiveMember`, `Balance`, `Geography_Germany`

### Evaluation Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| Accuracy | (TP+TN)/Total | General correctness |
| Precision | TP/(TP+FP) | Avoid false alarms |
| Recall | TP/(TP+FN) | Catch all churners |
| F1-Score | 2×P×R/(P+R) | Balance for imbalanced classes |
| ROC-AUC | Area under ROC curve | Overall discrimination |

### Target Performance

- F1-Score > 0.55
- ROC-AUC > 0.80
- Recall > 0.50

---

## 🛠️ Tech Stack

```
Python 3.8+
pandas, numpy
matplotlib, seaborn
scikit-learn
xgboost
Jupyter Notebook
```

**Install dependencies:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

---

## 📌 How to Run

1. Clone or download the repository
2. Install dependencies (see above)
3. Open the relevant Jupyter notebook for each TP
4. Run cells sequentially — each phase builds on the previous one

---

## 👤 Author

Zeineb — Data Engineering & Decisional Systems, ENET'Com Sfax
