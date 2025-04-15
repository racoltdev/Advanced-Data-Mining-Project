# CIS 530 Advanced Data Mining - Project
# Andrew Bajumpaa & Martin Mitrevski
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Load the dataset
enhanced_anxiety_data = pd.read_csv(r"C:\Users\marti\OneDrive\Desktop\Projects\Advanced Data Mining\enhanced_anxiety_dataset.csv")

df = pd.DataFrame(enhanced_anxiety_data)

# Separate features and target
X = df.drop(columns=["Anxiety Level (1-10)"])
y = df["Anxiety Level (1-10)"]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_cols),
        ("cat", OneHotEncoder(drop="first"), categorical_cols)
    ]
)

# Create a full pipeline with linear regression
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

mse, r2

# Full implementation of Somers' D for ordinal data based on the formula:
# D_Y|Y = (concordant - discordant) / total_pairs
# where total_pairs = concordant + discordant + ties in actual variable (Y)

# Reset indices for alignment
y_true_ord = y_test.reset_index(drop=True)
y_pred_ord = pd.Series(y_pred).reset_index(drop=True)

# Initialize counts
concordant = 0
discordant = 0
ties_in_true = 0

# Compare all pairs in the test set
n = len(y_true_ord)
for i in range(n):
    for j in range(i + 1, n):
        diff_true = y_true_ord[i] - y_true_ord[j]
        diff_pred = y_pred_ord[i] - y_pred_ord[j]
        product = diff_true * diff_pred
        if product > 0:
            concordant += 1
        elif product < 0:
            discordant += 1
        elif diff_true == 0:
            ties_in_true += 1

# Compute Somers' D
total_pairs = concordant + discordant + ties_in_true
somers_d_ord = (concordant - discordant) / total_pairs if total_pairs > 0 else None

{
    "Concordant": concordant,
    "Discordant": discordant,
    "Ties in Y (actual)": ties_in_true,
    "Total Pairs": total_pairs,
    "Somers' D (ordinal)": somers_d_ord
}

# Display results
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Somers' D : {somers_d_ord:.4f}")

