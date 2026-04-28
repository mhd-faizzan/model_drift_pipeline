import json
import logging
import os

import pandas as pd
import yaml
from scipy import stats

logger = logging.getLogger(__name__)


def load_training_distribution(config: dict) -> pd.DataFrame:
    """
    Loads training data and returns feature distributions as reference.

    Args:
        config: Full project config dict

    Returns:
        DataFrame of training features
    """
    df = pd.read_csv(config["data"]["raw_path"])

    # 2011 data only — this is what the model was trained on
    train_df = df[df["yr"] == 0].reset_index(drop=True)

    drop_cols = config["features"]["drop_cols"] + [config["features"]["target"]]
    feature_cols = [col for col in train_df.columns if col not in drop_cols]

    return train_df[feature_cols]


def load_live_window(config: dict) -> pd.DataFrame:
    """
    Loads the most recent N requests from the drift log.

    Args:
        config: Full project config dict

    Returns:
        DataFrame of recent live requests
    """
    log_path = config["drift"]["log_path"]

    if not os.path.exists(log_path):
        raise FileNotFoundError(
            f"Drift log not found at {log_path}. Run simulation first."
        )

    with open(log_path) as f:
        lines = f.readlines()

    if len(lines) < config["drift"]["window_size"]:
        raise ValueError(
            f"Not enough requests logged yet. "
            f"Need {config['drift']['window_size']}, got {len(lines)}."
        )

    # take the most recent window
    recent_lines = lines[-config["drift"]["window_size"]:]
    records = [json.loads(line) for line in recent_lines]
    df = pd.DataFrame(records)

    # drop prediction column — only compare features
    if "prediction" in df.columns:
        df = df.drop(columns=["prediction"])

    return df


def run_ks_test(
    train_df: pd.DataFrame,
    live_df: pd.DataFrame,
    config: dict
) -> dict:
    """
    Runs KS test on each feature comparing training vs live distribution.
    Drift is flagged only if the number of drifted features meets
    the min_drift_features threshold from config.

    Args:
        train_df: Training feature dataframe
        live_df: Live request feature dataframe
        config: Full project config dict

    Returns:
        Dict with per-feature results, drifted feature list, and overall drift flag
    """
    threshold = config["drift"]["p_value_threshold"]

    # default to 1 if not set — matches original behaviour
    min_drift_features = config["drift"].get("min_drift_features", 1)

    results = {}
    drifted_features = []

    for col in live_df.columns:
        if col not in train_df.columns:
            continue

        ks_stat, p_value = stats.ks_2samp(
            train_df[col].values,
            live_df[col].values
        )

        drifted = bool(p_value < threshold)

        if drifted:
            drifted_features.append(col)
            logger.warning(
                "Drift detected in '%s' — ks_stat: %.4f  p_value: %.4f",
                col, ks_stat, p_value
            )
        else:
            logger.info(
                "No drift in '%s' — ks_stat: %.4f  p_value: %.4f",
                col, ks_stat, p_value
            )

        results[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 4),
            "drifted": drifted
        }

    # only flag overall drift if enough features drifted
    drift_detected = len(drifted_features) >= min_drift_features

    if drift_detected:
        logger.warning(
            "%d feature(s) drifted (threshold: %d) — retraining should be triggered",
            len(drifted_features), min_drift_features
        )
    else:
        logger.info(
            "%d feature(s) drifted (threshold: %d) — model is healthy",
            len(drifted_features), min_drift_features
        )

    return {
        "drift_detected": bool(drift_detected),
        "drifted_features": drifted_features,
        "features": results,
    }


def save_drift_report(report: dict, config: dict) -> None:
    """
    Writes the latest drift report to report_path (overwrite)
    and appends to log_path (full history).

    Args:
        report: Drift report dict
        config: Full project config dict
    """
    os.makedirs("logs", exist_ok=True)

    # append to running log — full history
    log_path = config["drift"]["log_path"]
    with open(log_path, "a") as f:
        f.write(json.dumps(report) + "\n")
    logger.info("Drift log updated at %s", log_path)

    # overwrite latest report — this is what the PR body reads
    report_path = config["drift"]["report_path"]
    with open(report_path, "w") as f:
        f.write(json.dumps(report) + "\n")
    logger.info("Drift report saved to %s", report_path)


def run_drift_detection(config: dict) -> dict:
    """
    Full drift detection pipeline — loads data, runs KS test, saves report.

    Args:
        config: Full project config dict

    Returns:
        Drift report dict
    """
    logger.info("Loading training distribution...")
    train_df = load_training_distribution(config)

    logger.info(
        "Loading live window (%d requests)...", config["drift"]["window_size"]
    )
    live_df = load_live_window(config)

    logger.info("Running KS test on %d features...", len(live_df.columns))
    report = run_ks_test(train_df, live_df, config)

    save_drift_report(report, config)

    return report