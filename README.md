# Adult Income Prediction

A machine learning pipeline that predicts whether an adult's annual income exceeds **$50K** (**>50K** vs **<=50K**) from census data, featuring **XGBoost** optimization and export of test set predictions.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626)

---

## 📌 Overview

This project builds a binary classification pipeline on the Census Income (Adult) dataset. It covers exploratory data analysis, preprocessing, training and comparing three models, evaluating them on a held-out validation split, and generating predictions for the test set with the best-performing model (XGBoost).

**Primary evaluation metric:** macro F1-score.

## 📂 Dataset

The project uses the Census Income (Adult) dataset, provided as a training set (`train.csv`) and a test set (`test.csv`, which includes an `id` column and no target).

- **Target:** `income` → `<=50K` or `>50K`
- **Numerical features:** e.g. `age`, `hours-per-week`, `capital-gain`, `capital-loss`
- **Categorical features:** `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`

> Original source: [UCI Machine Learning Repository – Adult](https://archive.ics.uci.edu/dataset/2/adult)

## 🗂️ Repository Structure

```
adult-income-prediction/
├── data/        # train.csv and test.csv
├── notebook/    # adult_income_prediction.ipynb
├── output/      # predictions.csv
├── report/      # Technical report (.pdf)
├── .gitignore
└── README.md
```

## ⚙️ Pipeline

1. **Exploratory Data Analysis**
   - Data shape, types, summary statistics, and missing values, including `?` placeholders in categorical columns.
   - Target class distribution.
   - Correlation heatmap, histograms, boxplots, and a pair plot for numerical features.
   - Count plots of each categorical feature split by income class.
2. **Preprocessing**
   - Rows containing `?` or null values are removed from the training data.
   - Target labels are normalized to binary values (`<=50K` → 0, `>50K` → 1), also covering label variants with a trailing period.
   - Categorical features are one-hot encoded (`drop_first=True`).
   - Stratified 80/20 train/validation split (`random_state=42`).
   - Numerical features are standardized with `StandardScaler`, fitted on the training split only to avoid data leakage.
3. **Modeling**
   - **Logistic Regression** (multi-feature baseline)
   - **Random Forest** (100 trees, `class_weight='balanced'`)
   - **XGBoost** (400 estimators, `max_depth=8`, `learning_rate=0.1`, `scale_pos_weight` set from the class ratio to handle class imbalance)
4. **Evaluation**
   - Classification report and macro F1-score.
   - Confusion matrix, ROC curve, and precision-recall curve for each model.
5. **Prediction Export**
   - The test set is encoded and scaled with the same transformations as the training data, and its columns are aligned to the training feature set.
   - XGBoost predictions are saved to `output/predictions.csv` with columns `id` and `income`.

## 📊 Results

Macro F1-score on the held-out validation split (20% of the training data):

| Model | Macro F1 |
| --- | --- |
| Logistic Regression (multi-feature) | 0.7855 |
| Random Forest | 0.7902 |
| **XGBoost** | **0.8049** |

XGBoost achieved the best score and was used to generate the final test set predictions. Detailed analysis and conclusions are available in the technical report inside [`report/`](report/).

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/NajlaTsabita/adult-income-prediction.git
cd adult-income-prediction
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

### 4. Run the notebook

```bash
cd notebook
jupyter notebook adult_income_prediction.ipynb
```

The notebook reads data from `../data/` and writes predictions to `../output/`, so launch it from inside the `notebook/` folder. Run all cells in order (*Restart & Run All*) to reproduce the results.

## 🛠️ Tech Stack

- **Python**
- **pandas**, **NumPy** — data manipulation
- **scikit-learn** — preprocessing, Logistic Regression, Random Forest, evaluation
- **XGBoost** — main model
- **Matplotlib**, **Seaborn** — visualization
- **Jupyter Notebook** — development environment

## 👤 Author

**Najla Tsabita Afiyah**
GitHub: [@NajlaTsabita](https://github.com/NajlaTsabita)

---

⭐ If you find this repository useful, consider giving it a star!
