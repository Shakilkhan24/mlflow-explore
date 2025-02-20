import mlflow
import pandas as pd
import os

# Use the environment variable for the tracking URI
mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI'))

logged_model = 'runs:/bc6c201c2e1d49ec86c97a583ae3cbcc/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Convert input data into a pandas DataFrame
input_data = pd.DataFrame([[4.6, 3.6, 1.0, 0.2]], columns=["feature1", "feature2", "feature3", "feature4"])

# Make prediction
pred = loaded_model.predict(input_data)
print(pred)
