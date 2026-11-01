# XGBoost: Learning From Mistakes — Campus Placement Prediction

Predicting whether an MBA student gets **placed** in a job using **XGBoost**, with leakage-safe preprocessing and a tuned decision threshold.

This repository has the project notebook and the presentation that explains XGBoost from scratch: boosting intuition, the math, our pipeline, and an honest evaluation.

---

## 📌 Project Overview

| | |
|---|---|
| **Task** | Binary classification: Placed (1) vs Not Placed (0) |
| **Dataset** | [Factors Affecting Campus Placement](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement) (Kaggle) |
| **Model** | `xgboost.XGBClassifier` |
| **Key idea** | Choosing the decision threshold on validation data turned a model that predicted "Placed" for everyone into one that finds 10 of 13 non-placed students |

---

## 📊 Dataset

- **215** MBA students, **15** original columns
- **12** predictive features after removing:
  - `sl_no`: a row ID with no predictive value
  - `salary`: only recorded for placed students, so it would leak the target
- Target balance: **~69% Placed / ~31% Not Placed**
- No missing values (after dropping `salary`) and no duplicate rows

**Features:** `gender`, `ssc_p`, `ssc_b`, `hsc_p`, `hsc_b`, `hsc_s`, `degree_p`, `degree_t`, `workex`, `etest_p`, `specialisation`, `mba_p`

---

## 🔍 Key EDA Findings

| Finding | Value |
|---|---|
| Correlation with placement: `ssc_p` (10th grade) | **0.608** |
| Correlation: `hsc_p` (12th grade) | 0.491 |
| Correlation: `degree_p` | 0.480 |
| Correlation: `etest_p` / `mba_p` | 0.128 / 0.077 |
| Placement rate: work experience Yes vs No | ≈86% vs ≈60% |
| Placement rate: Mkt&Fin vs Mkt&HR | ≈79% vs ≈56% |

Earlier academic performance is more strongly associated with placement than the MBA score or the employability test. These are associations in this dataset, not causes.

---

## ⚙️ Methodology

```
Dataset → Leakage removal → Stratified split → Preprocessing → XGBoost → Validation
```

1. **Split** (stratified, `random_state=42`): Train **137** · Validation **35** · Test **43**
2. **Preprocessing** (`ColumnTransformer`, fitted on the training set only):
   - `StandardScaler` for numeric columns (not required for trees, but harmless)
   - `OneHotEncoder(handle_unknown="ignore")` for categorical columns
3. **Model configuration**

   | Hyperparameter | Value |
   |---|---|
   | n_estimators | 2000 (upper bound) |
   | learning_rate | 0.05 |
   | max_depth | 3 |
   | min_child_weight | 3 |
   | subsample | 0.8 |
   | colsample_bytree | 0.8 |
   | reg_lambda | 1.0 |
   | tree_method | hist |
   | eval_metric | auc |
   | early_stopping_rounds | 100 |

   Early stopping selected **best iteration = 3**.

4. **Threshold tuning:** we scanned thresholds from 0.10 to 0.95 on the **validation** set and picked the best macro F1, which was **0.70** (macro F1 = 0.873, balanced accuracy = 0.892).

---

## 📈 Results (Test Set, 43 students)

| Metric | Threshold 0.5 | Threshold 0.7 |
|---|---|---|
| Accuracy | 0.70 | 0.65 |
| Macro F1 | 0.41 | **0.64** |
| Recall: Not Placed | 0.00 | **0.77** |
| Recall: Placed | 1.00 | 0.60 |
| Precision: Placed | 0.70 | **0.86** |
| ROC-AUC | 0.827 | 0.827 |

**Confusion matrices** (rows = actual, columns = predicted, order [Not Placed, Placed]):

```
Threshold 0.5        Threshold 0.7
[[ 0, 13],           [[10,  3],
 [ 0, 30]]            [12, 18]]
```

> Higher accuracy does not always mean a better model for the purpose. Changing the threshold changes the labels, not the ranking, so ROC-AUC stays the same.

⚠️ **Limitation:** the test set has only 43 students, so results may change noticeably with a different split. This is an educational project, not a deployment-ready tool.

---

## 📁 Repository Structure

```
xgboost-campus-placement/
├── Campus_Recruitment.ipynb      # EDA, preprocessing, training, evaluation
├── presentation/
│   └── XGBoost_Learning_From_Mistakes.pptx
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
git clone https://github.com/<your-username>/xgboost-campus-placement.git
cd xgboost-campus-placement
pip install -r requirements.txt
jupyter notebook Campus_Recruitment.ipynb
```

`requirements.txt`:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
kagglehub
tqdm
jupyter
```

The notebook downloads the dataset automatically with `kagglehub`. You may need a Kaggle account and API token.

---

## 👥 Team

| Name | GitHub |
|---|---|
| Fahad Al-Nahi | [@Skkinner](https://github.com/Skkinner) |
| [Member 2] | [@username](https://github.com/username) |
| [Member 3] | [@username](https://github.com/username) |
| [Member 4] | [@username](https://github.com/username) |

---

## 🙏 Acknowledgments

- Dataset: Ben Roshan D, *Factors Affecting Campus Placement*, Kaggle
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD '16
- [XGBoost documentation](https://xgboost.readthedocs.io/)
