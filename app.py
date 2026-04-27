import json
import logging
import os
import pickle

import pandas as pd
import yaml
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI()

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

with open(config["model"]["save_path"], "rb") as f:
    model = pickle.load(f)

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

    # log incoming features for drift detection
    log_entry = request.model_dump()
    log_entry["prediction"] = float(prediction)

    with open(config["drift"]["log_path"], "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    logger.info("Prediction: %.2f", prediction)

    return {"predicted_count": round(float(prediction), 2)}


@app.get("/drift-report")
def drift_report() -> dict:
    log_path = config["drift"]["log_path"]

    if not os.path.exists(log_path):
        return {"message": "No requests logged yet"}

    with open(log_path) as f:
        lines = f.readlines()

    return {"total_requests_logged": len(lines)}