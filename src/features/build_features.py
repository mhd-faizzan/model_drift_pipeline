import logging
import pandas as pd

logger = logging.getLogger(__name__)


def build_features(df: pd.DataFrame, target: str, drop_cols: list) -> tuple[pd.DataFrame, pd.Series]:
    """
    Drops irrelevant columns and separates features from target.

    Args:
        df: Raw dataframe
        target: Target column name
        drop_cols: Columns to drop before training

    Returns:
        Tuple of (X, y)
    """
    cols_to_drop = [col for col in drop_cols if col in df.columns]
    cols_to_drop.append(target)

    X = df.drop(columns=cols_to_drop)
    y = df[target]

    logger.info("Features shape: %s", X.shape)
    logger.info("Target shape: %s", y.shape)
    logger.info("Feature columns: %s", X.columns.tolist())

    return X, y