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

    # drop prediction column — we only compare features
    if "prediction" in df.columns:
        df = df.drop(columns=["prediction"])

    return df


def run_ks_test(train_df: pd.DataFrame, live_df: pd.DataFrame, threshold: float) -> dict:
    """
    Runs KS test on each feature comparing training vs live distribution.

    Args:
        train_df: Training feature dataframe
        live_df: Live request feature dataframe
        threshold: p-value threshold below which drift is flagged

    Returns:
        Dict with per-feature results and overall drift flag
    """
    results = {}
    drift_detected = False

    for col in live_df.columns:
        if col not in train_df.columns:
            continue

        ks_stat, p_value = stats.ks_2samp(
            train_df[col].values,
            live_df[col].values
        )

        drifted = bool(p_value < threshold)

        if drifted:
            drift_detected = True

        results[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 4),
            "drifted": drifted
        }

        if drifted:
            logger.warning(
                "Drift detected in '%s' — p_value: %.4f", col, p_value
            )
        else:
            logger.info("No drift in '%s' — p_value: %.4f", col, p_value)

    return {
        "drift_detected": bool(drift_detected),
        "features": results
    }


def save_drift_report(report: dict, config: dict) -> None:
    """
    Appends drift report to drift log file.

    Args:
        report: Drift report dict
        config: Full project config dict
    """
    os.makedirs("logs", exist_ok=True)
    report_path = "logs/drift_report.jsonl"

    with open(report_path, "a") as f:
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
    report = run_ks_test(train_df, live_df, config["drift"]["p_value_threshold"])

    save_drift_report(report, config)

    if report["drift_detected"]:
        logger.warning("DRIFT DETECTED — retraining should be triggered")
    else:
        logger.info("No drift detected — model is healthy")

    return report