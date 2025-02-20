# [NOTE] imports 

import mlflow
import mlflow.xgboost
import optuna
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from optuna.integration import MLflowCallback
from mlflow.models import infer_signature

# [NOTE] Loading data and preprocessing
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# [NOTE] Define objective function
def objective(trial):
    param = {
        'verbosity': 0,
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
    }
   
    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
   
    accuracy = accuracy_score(y_test, preds)
    return accuracy

# [NOTE] Callback for Optuna MLflow integration
mlflow_callback = MLflowCallback(
    tracking_uri=mlflow.get_tracking_uri(),
    metric_name="accuracy"
)

if __name__ == "__main__":
    mlflow.set_experiment("XGBoost-Optuna-Hyper-parameters")  # Set your MLflow experiment name
    study = optuna.create_study(direction="maximize", study_name="xgboost_optuna_study")  # [NOTE] Study name would appear in mlflow server
    study.optimize(objective, n_trials=20, callbacks=[mlflow_callback])  # [NOTE] Callback for Optuna MLflow integration
   
    best_trial = study.best_trial
    print(f"Best trial: {best_trial.number}")
    print(f"Best accuracy: {best_trial.value}")
    print(f"Best hyperparameters: {best_trial.params}")
   
    with mlflow.start_run(run_name="Best_Model"):
        mlflow.log_params(best_trial.params)
        mlflow.log_metric("accuracy", best_trial.value)
       
        best_model = xgb.XGBClassifier(**best_trial.params)
        best_model.fit(X_train, y_train)
       
        # Infer the model signature
        y_pred = best_model.predict(X_train)
        signature = infer_signature(X_train, y_pred)
       
        # Log the model with signature
        model_info = mlflow.xgboost.log_model(
            xgb_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=X_train[:5],
            registered_model_name="XGBoost-Iris-Classifier"
        )
       
        print(f"Logged best model with params: {best_trial.params} and accuracy: {best_trial.value}")
        print(f"Model logged as: {model_info.model_uri}")
        print(f"Model registered as: {model_info.model_uri.split('/')[-1]}")