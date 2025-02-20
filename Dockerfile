FROM python:3.9-slim 

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY mlruns/821402284970385508/bc6c201c2e1d49ec86c97a583ae3cbcc/artifacts/model /app/model
COPY mlruns/821402284970385508/bc6c201c2e1d49ec86c97a583ae3cbcc/metrics /app/metrics
COPY mlruns/821402284970385508/bc6c201c2e1d49ec86c97a583ae3cbcc/params /app/params
COPY mlruns/821402284970385508/bc6c201c2e1d49ec86c97a583ae3cbcc/tags /app/tags
COPY mlruns/821402284970385508/bc6c201c2e1d49ec86c97a583ae3cbcc/meta.yaml /app/meta.yaml

COPY OPTUNA_MLFLOW_BEST_MODEL.py .

ENV MLFLOW_TRACKING_URI=file:///app/mlruns

CMD ["python", "OPTUNA_MLFLOW_BEST_MODEL_VERSION.py"]


