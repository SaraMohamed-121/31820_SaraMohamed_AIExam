# Regression Task: California Housing Prices Prediction

## Objective

The goal of this project is to build a regression pipeline that predicts the **median house value** from structured/tabular data. The project implements, compares, and evaluates multiple machine learning models to determine the best-performing solution.

---

## 🗂 Dataset

**California Housing Prices** – Kaggle
[Link to dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

* **Samples:** 20,640
* **Features:** 10 (e.g., median_income, house age, total rooms, population, ocean proximity)
* **Target:** `median_house_value`

---

## Project Structure

```
regression_task/
├── data/
│   └── housing.csv             # Raw dataset
├── eda.py                      # Exploratory Data Analysis
├── preprocess.py               # Data cleaning & feature engineering
├── train.py                    # Model training & hyperparameter tuning
├── evaluate.py                 # Model evaluation & visualizations
├── outputs/
│   ├── eda_plots/              # EDA figures
│   ├── model_comparison.png    # R² comparison plot
│   ├── actual_vs_predicted.png # Predicted vs Actual plot
│   ├── residuals.png           # Residual distribution plot
│   ├── feature_importance.png  # Top features
│   └── *.pkl                   # Saved trained models
└── README.md
```

---


## Models Trained

* **Linear Regression** – Baseline
* **Random Forest Regressor** – Tree-based model
* **XGBoost Regressor** – Gradient boosting model

---

## How to Run

1. Clone repository and download dataset into `data/housing.csv`

2. **Install Requirements**

   Before running any scripts, install all required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

   This will install all necessary libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `joblib`) with compatible versions.

3. Run EDA:

   ```bash
   python eda.py
   ```

4. Build preprocessing and train models:

   ```bash
   python train.py
   ```

5. Evaluate models and generate visualizations:

   ```bash
   python evaluate.py
   ```

6. Trained models are saved in `outputs/` for future use.

