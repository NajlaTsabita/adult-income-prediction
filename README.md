# Adult Income Prediction

An end-to-end machine learning pipeline that predicts whether an adult's annual income exceeds **$50K** (**>50K** vs **<=50K**) from census data, featuring **XGBoost** optimization and export of test set predictions.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626)

---

## 📌 Overview

This project builds a reproducible binary classification pipeline on the Census Income (Adult) dataset. It covers exploratory data analysis, preprocessing, model training and comparison, hyperparameter tuning, evaluation, and generating predictions for the test set.

**Primary evaluation metric:** macro F1-score.

## 📂 Dataset

The project uses the Census Income (Adult) dataset, split into training and test sets.

- **Target:** `income` → `<=50K` or `>50K`
- **Features:** a mix of numerical (e.g. `age`, `hours-per-week`, `capital-gain`, `capital-loss`) and categorical variables (e.g. `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`)

> Original source: [UCI Machine Learning Repository – Adult](https://archive.ics.uci.edu/dataset/2/adult)

## 🗂️ Repository Structure

```
adult-income-prediction/
├── data/        # Training and test datasets
├── notebook/    # Jupyter notebook: EDA, preprocessing, modeling, evaluation
├── output/      # Exported test set predictions (.csv)
├── report/      # Technical report (.pdf)
├── .gitignore
└── README.md
```

## ⚙️ Pipeline

1. **Exploratory Data Analysis**
   - Data types, unique values of categorical columns, and target distribution.
   - Detection of missing or invalid values (e.g. `?`) in columns such as `workclass`, `occupation`, and `native-country`.
   - Income patterns across education, occupation, working hours, and capital gain.
2. **Preprocessing**
   - Missing value handling.
   - One-hot encoding of categorical features, applied consistently to train and test data so both share the same columns.
   - Scaling of numerical features, with the scaler fitted on the training data only to avoid data leakage.
3. **Modeling**
   - Baselines: Logistic Regression and Random Forest.
   - Main model: **XGBoost**.
4. **Optimization**
   - Hyperparameter tuning of XGBoost with cross-validation.
5. **Evaluation**
   - Macro F1-score, classification report, and confusion matrix on validation data.
6. **Prediction Export**
   - Predictions for the test set are saved as a CSV file in `output/`.

## 📊 Results

| Model | Macro F1 (Validation) |
| --- | --- |
| Logistic Regression | _TBD_ |
| Random Forest | _TBD_ |
| XGBoost (default) | _TBD_ |
| **XGBoost (tuned)** | **_TBD_** |

Detailed analysis and conclusions are available in the technical report inside [`report/`](report/).

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
jupyter notebook notebook/
```

Make sure the dataset files are placed in the `data/` folder, then run all cells in order (*Restart & Run All*). The prediction file will be generated in `output/`.

## 🛠️ Tech Stack

- **Python**
- **pandas**, **NumPy** — data manipulation
- **scikit-learn** — preprocessing, baseline models, evaluation
- **XGBoost** — main model
- **Matplotlib**, **Seaborn** — visualization
- **Jupyter Notebook** — development environment

## 👤 Author

GitHub: [@NajlaTsabita](https://github.com/NajlaTsabita)

---

⭐ If you find this repository useful, consider giving it a star!
