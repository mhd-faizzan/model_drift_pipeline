import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict
) -> tuple[RandomForestRegressor, dict]:
    """
    Trains a RandomForest regressor on the provided features.

    Args:
        X: Feature dataframe
        y: Target series
        config: Full project config dict

    Returns:
        Tuple of trained model and evaluation metrics dict
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )

    model = RandomForestRegressor(
        n_estimators=config["model"]["n_estimators"],
        random_state=config["model"]["random_state"],
        n_jobs=config["model"].get("n_jobs", -1),
    )

    model.fit(X_train, y_train)
    logger.info("Model trained on %d rows", len(X_train))

    metrics = evaluate_model(model, X_val, y_val)

    return model, metrics


def evaluate_model(
    model: RandomForestRegressor,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> dict:
    """
    Evaluates model on validation set and returns metrics.

    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation target

    Returns:
        Dict with mae, rmse, and r2
    """
    predictions = model.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_val, predictions)))
    r2 = r2_score(y_val, predictions)

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }

    logger.info("Validation MAE:  %.4f", mae)
    logger.info("Validation RMSE: %.4f", rmse)
    logger.info("Validation R2:   %.4f", r2)

    return metrics


def save_model(
    model: RandomForestRegressor,
    metrics: dict,
    config: dict
) -> None:
    """
    Saves trained model and metrics to disk.

    Args:
        model: Trained model
        metrics: Evaluation metrics dict
        config: Full project config dict
    """
    os.makedirs(os.path.dirname(config["model"]["save_path"]), exist_ok=True)

    with open(config["model"]["save_path"], "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", config["model"]["save_path"])

    with open(config["model"]["metrics_path"], "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", config["model"]["metrics_path"])