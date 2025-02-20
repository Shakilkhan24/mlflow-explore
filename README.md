# XGBoost Hyperparameter Optimization with Optuna and MLflow

## Overview
This project demonstrates hyperparameter optimization for an XGBoost classifier using Optuna while tracking experiments with MLflow. The dataset used is the Iris dataset from scikit-learn.

## Features
- Hyperparameter tuning using **Optuna**.
- Tracking experiment results using **MLflow**.
- Logging best model parameters and metrics.
- Registering the trained model in MLflow for future use.

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install mlflow optuna xgboost scikit-learn
```

## Dataset
The Iris dataset is used, which consists of three classes of iris flowers with four features each. The dataset is loaded using `sklearn.datasets.load_iris`.

## Code Structure
1. **Loading Data:**
   - The dataset is split into training and testing sets (80% training, 20% testing).
2. **Objective Function:**
   - Defines the XGBoost classifier with hyperparameters suggested by Optuna.
   - Trains the model and evaluates accuracy on the test set.
3. **Hyperparameter Optimization:**
   - Runs Optuna's `create_study` method with a direction to maximize accuracy.
   - Uses MLflow callback to log results automatically.
4. **Logging Best Model to MLflow:**
   - Logs best hyperparameters and accuracy to MLflow.
   - Trains the best model and logs it with an inferred signature.
   - Registers the model as `XGBoost-Iris-Classifier` in MLflow.

## Running the Script
Execute the script using:

```bash
python OPTUNA_MLFLOW.py
```

## Expected Output
- Optuna will perform `n_trials=20` to find the best hyperparameters.
- The best hyperparameters and accuracy will be printed.
- The best model will be saved and registered in MLflow.

## Viewing MLflow Results
To visualize the tracked experiments, start the MLflow UI:

```bash
mlflow ui
```

Then, open [http://localhost:5000](http://localhost:5000) in your browser to explore logged experiments.


## After the training and monitoring the logs in mlruns directory , 
## You can load the best or any model from there for inferencing 

```py
logged_model = 'runs:/bc6c201c2e1d49ec86c97a583ae3cbcc/model'
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)
```
Then you can you use the loaded model for inferencing on preprocessed test data.


## License
This project is licensed under the MIT License.

