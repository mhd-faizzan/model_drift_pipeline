import logging
import json
import os
import shutil

import yaml

from src.data.load_data import load_raw_data, split_by_year
from src.features.build_features import build_features
from src.models.train import train_model, save_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def backup_champion_metrics(config: dict) -> None:
    """
    Copies current metrics.json to champion_metrics.json before retraining.
    Skips silently if no existing metrics found — handles first run.

    Args:
        config: Full project config dict
    """
    metrics_path = config["model"]["metrics_path"]
    champion_path = config["model"].get("champion_metrics_path")

    if not champion_path:
        return

    if os.path.exists(metrics_path):
        os.makedirs(os.path.dirname(champion_path), exist_ok=True)
        shutil.copy2(metrics_path, champion_path)
        logger.info("Champion metrics backed up to %s", champion_path)
    else:
        logger.info("No existing metrics found — skipping champion backup")


def main() -> None:
    """
    Runs the full training pipeline:
    1. Load config
    2. Back up champion metrics
    3. Load and split raw data
    4. Build features
    5. Train model
    6. Save model and metrics
    """
    logger.info("Starting training pipeline")

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    # back up before anything gets overwritten
    backup_champion_metrics(config)

    df = load_raw_data(config["data"]["raw_path"])

    train_df, _ = split_by_year(
        df,
        config["data"]["train_year"],
        config["data"]["live_year"]
    )

    X, y = build_features(
        train_df,
        config["features"]["target"],
        config["features"]["drop_cols"]
    )

    model, metrics = train_model(X, y, config)

    save_model(model, metrics, config)

    logger.info("Training complete. Metrics: %s", metrics)


if __name__ == "__main__":
    main()