import mlflow
from mlflow.models import infer_signature

import mlflow.sklearn
import mlflow.xgboost
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Load and split the data
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models with their parameters
models = {
    "SVM": SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
}

# Set the experiment name
mlflow.set_experiment("Iris_Classification_Models")

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log parameters
        params = model.get_params()
        mlflow.log_params(params)
        
        # Log metric
        mlflow.log_metric("accuracy", accuracy)
        
        # [NOTE] Infer the model signature 
        signature = infer_signature(X_train, y_pred)
        
        # [NOTE] Log model with signature and REGISTER it [IMPORTANT]
        if isinstance(model, xgb.XGBClassifier):
            model_info = mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="model",
                signature=signature,
                input_example=X_train[:5],
                registered_model_name=f"Iris-{model_name}"
            )
        else:
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=X_train[:5],
                registered_model_name=f"Iris-{model_name}"
            )
        
        print(f"{model_name} - Accuracy: {accuracy:.4f}")
        print(f"Model logged as: {model_info.model_uri}")
        print(f"Model registered as: {model_info.model_uri.split('/')[-1]}")

print("All models have been trained, logged, and registered in MLflow.")

