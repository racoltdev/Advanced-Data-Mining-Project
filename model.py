# CIS 530 Advanced Data Mining - Project
# Andrew Bajumpaa & Martin Mitrevski
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import metrics

def mlr(data_file):
	# Load the dataset
	enhanced_anxiety_data = pd.read_csv(data_file)

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
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	# Train the model
	pipeline.fit(X_train, y_train)

	y_pred = pipeline.predict(X_test)
	Metrics = metrics.Metrics(y_test, y_pred)
	Metrics.print()
