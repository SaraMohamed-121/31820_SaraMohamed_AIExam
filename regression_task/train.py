import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

from preprocess import load_and_split_data, build_preprocessing_pipeline

# Load data
X_train, X_test, y_train, y_test = load_and_split_data()
preprocessor = build_preprocessing_pipeline(X_train)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest":RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        random_state=42,
        eval_metric="rmse",
        use_label_encoder=False
    )
}

trained_models = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    pipeline.fit(X_train, y_train)
    trained_models[name] = pipeline
    joblib.dump(pipeline, f"outputs/{name.replace(' ', '_')}.pkl")

print("Models trained successfully.")

# Hyperparameter tuning for best model (XGBoost)
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.01, 0.1, 0.2]
}

xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(
        random_state=42,
        eval_metric="rmse",
        use_label_encoder=False
    ))
])

search = RandomizedSearchCV(
    xgb_pipeline,
    param_grid,
    cv=3,
    scoring="r2",
    n_iter=5,
    random_state=42
)

search.fit(X_train, y_train)

joblib.dump(search.best_estimator_, "outputs/best_model.pkl")

print("Best model saved.")