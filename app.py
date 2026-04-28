import json
import logging
import os
import pickle

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI()

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

model_path = config["model"]["save_path"]

if not os.path.exists(model_path):
    raise RuntimeError(
        f"Model not found at {model_path}. Run main.py to train first."
    )

with open(model_path, "rb") as f:
    model = pickle.load(f)

logger.info("Model loaded from %s", model_path)

os.makedirs("logs", exist_ok=True)


class PredictionRequest(BaseModel):
    season: int
    mnth: int
    hr: int
    holiday: int
    weekday: int
    workingday: int
    weathersit: int
    temp: float
    atemp: float
    hum: float
    windspeed: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    features = pd.DataFrame([request.model_dump()])
    prediction = model.predict(features)[0]

    # log incoming features and prediction for drift detection
    log_entry = request.model_dump()
    log_entry["prediction"] = float(prediction)

    with open(config["drift"]["log_path"], "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    logger.info("Prediction: %.2f", prediction)

    return {"predicted_count": round(float(prediction), 2)}


@app.get("/drift-report")
def drift_report() -> dict:
    """
    Returns latest drift report if available, otherwise returns
    total requests logged so far.
    """
    report_path = config["drift"]["report_path"]
    log_path = config["drift"]["log_path"]

    # return latest drift report if it exists
    if os.path.exists(report_path):
        with open(report_path) as f:
            lines = f.readlines()
        if lines:
            latest = json.loads(lines[-1])
            total_logged = 0
            if os.path.exists(log_path):
                with open(log_path) as f:
                    total_logged = sum(1 for _ in f)
            return {
                "total_requests_logged": total_logged,
                "latest_drift_report": latest,
            }

    # drift detection has not run yet — return request count only
    if not os.path.exists(log_path):
        return {"message": "No requests logged yet"}

    with open(log_path) as f:
        total_logged = sum(1 for _ in f)

    return {
        "total_requests_logged": total_logged,
        "latest_drift_report": None,
    }