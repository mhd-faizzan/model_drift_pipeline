import logging
import yaml

from src.data.load_data import load_raw_data, split_by_year
from src.features.build_features import build_features
from src.models.train import train_model, save_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting training pipeline")

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

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