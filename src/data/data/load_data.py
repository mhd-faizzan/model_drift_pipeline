import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Loads raw bike sharing CSV from disk.

    Args:
        path: Path to hour.csv

    Returns:
        Raw dataframe with all original columns

    Example:
        load_raw_data("data/raw/hour.csv")
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found at {path}. Download hour.csv from UCI first."
        )

    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def split_by_year(df: pd.DataFrame, train_year: int, live_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into train and live sets by year.

    Args:
        df: Full raw dataframe
        train_year: Year to use for training (2011)
        live_year: Year to use for simulation (2012)

    Returns:
        Tuple of (train_df, live_df)
    """
    train_df = df[df["yr"] == 0].reset_index(drop=True)
    live_df = df[df["yr"] == 1].reset_index(drop=True)

    logger.info("Train set: %d rows (year %d)", len(train_df), train_year)
    logger.info("Live set: %d rows (year %d)", len(live_df), live_year)

    return train_df, live_df