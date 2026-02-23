import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from preprocess import load_and_split_data

# Load data
X_train, X_test, y_train, y_test = load_and_split_data()

# Load models
model_files = {
    "Linear Regression": "outputs/Linear_Regression.pkl",
    "Random Forest": "outputs/Random_Forest.pkl",
    "XGBoost": "outputs/XGBoost.pkl"
}

results = []

for name, path in model_files.items():
    model = joblib.load(path)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results.append([name, mae, mse, rmse, r2])

results_df = pd.DataFrame(results, columns=["Model", "MAE", "MSE", "RMSE", "R2"])
print(results_df)

# Save comparison plot
results_df.set_index("Model")["R2"].plot(kind="bar")
plt.title("Model Comparison (R2 Score)")
plt.savefig("outputs/model_comparison.png")
plt.close()

# Evaluate best model
best_model = joblib.load("outputs/best_model.pkl")
y_pred = best_model.predict(X_test)

# Actual vs Predicted
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.savefig("outputs/actual_vs_predicted.png")
plt.close()

# Residuals
residuals = y_test - y_pred
plt.figure(figsize=(6,6))
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.savefig("outputs/residuals.png")
plt.close()

# Feature Importance (if tree-based model)
model = best_model.named_steps["model"]
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    plt.figure(figsize=(8,5))
    plt.bar(range(len(importances)), importances)
    plt.title("Feature Importance")
    plt.savefig("outputs/feature_importance.png")
    plt.close()