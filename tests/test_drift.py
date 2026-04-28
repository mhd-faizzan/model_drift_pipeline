import json
import os
import tempfile

import pandas as pd
import pytest

from src.models.detect_drift import run_ks_test


def test_no_drift_same_distribution():
    # same distribution should not trigger drift
    train_df = pd.DataFrame({"temp": [0.5, 0.6, 0.5, 0.6, 0.5]})
    live_df = pd.DataFrame({"temp": [0.5, 0.6, 0.5, 0.6, 0.5]})

    report = run_ks_test(train_df, live_df, threshold=0.05)

    assert report["drift_detected"] is False
    assert report["features"]["temp"]["drifted"] is False


def test_drift_detected_different_distribution():
    # completely different distributions should trigger drift
    train_df = pd.DataFrame({"temp": [0.8, 0.9, 0.8, 0.9, 0.8] * 20})
    live_df = pd.DataFrame({"temp": [0.1, 0.2, 0.1, 0.2, 0.1] * 20})

    report = run_ks_test(train_df, live_df, threshold=0.05)

    assert report["drift_detected"] is True
    assert report["features"]["temp"]["drifted"] is True


def test_ks_report_structure():
    train_df = pd.DataFrame({"temp": [0.5, 0.6], "hum": [0.4, 0.5]})
    live_df = pd.DataFrame({"temp": [0.5, 0.6], "hum": [0.4, 0.5]})

    report = run_ks_test(train_df, live_df, threshold=0.05)

    assert "drift_detected" in report
    assert "features" in report
    assert "temp" in report["features"]
    assert "ks_statistic" in report["features"]["temp"]
    assert "p_value" in report["features"]["temp"]
    assert "drifted" in report["features"]["temp"]