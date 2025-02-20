import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set the MLflow tracking 
mlflow.set_tracking_uri("http://34.13.195.239:5000")

# Set the experiment name or create one if it doesn't exist
mlflow.set_experiment('test_experiment')

# Load data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Define model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Start logging the experiment
with mlflow.start_run():

    # Log hyperparameters
    mlflow.log_param('n_estimators', 100)

    # Fit model
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric('accuracy', acc)

    # Log model
    mlflow.sklearn.log_model(model, 'model')

