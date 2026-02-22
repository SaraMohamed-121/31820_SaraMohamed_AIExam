import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create output directory
os.makedirs("outputs/eda_plots", exist_ok=True)

# Load dataset
df = pd.read_csv("data/housing.csv")

print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())

# 1. Target Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["median_house_value"], bins=50, kde=True)
plt.title("Distribution of Median House Value")
plt.savefig("outputs/eda_plots/target_distribution.png")
plt.close()

# 2. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.savefig("outputs/eda_plots/correlation_heatmap.png")
plt.close()

# 3. Median Income vs House Value
plt.figure(figsize=(8, 5))
sns.scatterplot(x="median_income", y="median_house_value", data=df)
plt.title("Median Income vs House Value")
plt.savefig("outputs/eda_plots/income_vs_value.png")
plt.close()

# 4. Boxplot for Outliers
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["median_house_value"])
plt.title("Boxplot of Median House Value")
plt.savefig("outputs/eda_plots/boxplot_target.png")
plt.close()