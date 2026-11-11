"""
WEEK 1 PROJECT: Python for Machine Learning & Data Preprocessing
Dataset: student_performance_raw.csv (synthetic practice data)

Install dependencies:
    pip install pandas numpy matplotlib scikit-learn

Run:
    python preprocessing_project.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

DATA_FILE = Path(__file__).with_name("student_performance_raw.csv")
OUTPUT_DIR = Path(__file__).with_name("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Load data
df = pd.read_csv(DATA_FILE)
print("\\n1. DATA LOADED")
print(df.head())
print("\\nShape:", df.shape)

# 2. Initial inspection and EDA
print("\\n2. DATA INFORMATION")
print(df.info())
print("\\nSummary statistics:")
print(df.describe(include="all"))
print("\\nMissing values:")
print(df.isnull().sum())
print("\\nDuplicate rows:", df.duplicated().sum())

# Save an EDA chart
df["Final_Score"].plot(kind="hist", bins=8, edgecolor="black",
                       title="Distribution of Final Scores")
plt.xlabel("Final score")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "final_score_distribution.png")
plt.close()

# 3. Remove duplicate records
df = df.drop_duplicates()

# 4. Handle missing values
# Median is used for numeric columns because it is less affected by extreme values.
df["Study_Hours"] = df["Study_Hours"].fillna(df["Study_Hours"].median())
df["Attendance_Percent"] = df["Attendance_Percent"].fillna(df["Attendance_Percent"].median())
# Mode is used for categorical columns if any missing values are present.
for column in ["Gender", "Area", "Internet_Access"]:
    if df[column].isnull().any():
        df[column] = df[column].fillna(df[column].mode()[0])

# 5. Feature selection
# Student_ID is an identifier, so it is not useful as a predictive feature.
# Final_Score is the target variable.
selected_features = [
    "Age", "Gender", "Area", "Study_Hours",
    "Attendance_Percent", "Internet_Access"
]
X = df[selected_features].copy()
y = df["Final_Score"].copy()

# 6. Encode categorical variables using one-hot encoding
X_encoded = pd.get_dummies(
    X,
    columns=["Gender", "Area", "Internet_Access"],
    drop_first=True,
    dtype=int
)

# 7. Normalize numerical features using Min-Max scaling
numeric_columns = ["Age", "Study_Hours", "Attendance_Percent"]
scaler = MinMaxScaler()
X_encoded[numeric_columns] = scaler.fit_transform(X_encoded[numeric_columns])

# 8. Combine processed features and target, then save
processed = X_encoded.copy()
processed["Final_Score"] = y.values
processed.to_csv(OUTPUT_DIR / "student_performance_cleaned.csv", index=False)

# Save a simple correlation chart for numeric columns
correlation = df[["Age", "Study_Hours", "Attendance_Percent", "Final_Score"]].corr()
correlation.to_csv(OUTPUT_DIR / "numeric_correlation.csv")

print("\\n3. PREPROCESSING COMPLETE")
print("Processed data shape:", processed.shape)
print("\\nProcessed data preview:")
print(processed.head())
print("\\nFiles saved in:", OUTPUT_DIR.resolve())
