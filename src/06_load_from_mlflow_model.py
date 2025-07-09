import mlflow
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. Loading data
# TODO: Replace with the path to your dataset
print("Loading data...")
data = pd.read_csv("data/fake_data.csv")
X = data.drop(columns=["date", "demand"])
X = X.astype('float')

# 2. Define the path to the MLflow model
# TODO: Replace with the path to your "rf_apples" folder created previously
model_path = '/home/ubuntu/MLflow_Course/mlruns/864063752250630287/4bdc560e8b47457c806ad22d3cdcc694/artifacts/rf_apples'  # For example: '/home/ubuntu/MLflow/mlruns/EXPERIMENT_ID/RUN_ID/artifacts/rf_apples'

# 3. Load the model
print("Loading model...")
model = mlflow.sklearn.load_model(model_path)

# 4. Make predictions on the entire dataset
print("Calculating predictions...")
predictions = model.predict(X)

# 5. Calculate and display the average of predictions
# TODO: Calculate the mean of predictions
mean_prediction = predictions.mean()  # Use the appropriate numpy or pandas function

print(f"\nResults:")
print(f"Number of predictions: {len(predictions)}")
print(f"Mean of predictions: {mean_prediction:.2f}")