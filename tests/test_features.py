import pandas as pd
import pytest

from src.features.build_features import build_features


def test_build_features_drops_target():
    df = pd.DataFrame({
        "temp": [0.5, 0.6],
        "hum": [0.4, 0.5],
        "cnt": [100, 200]
    })

    X, y = build_features(df, target="cnt", drop_cols=[])

    assert "cnt" not in X.columns
    assert len(y) == 2


def test_build_features_drops_specified_cols():
    df = pd.DataFrame({
        "instant": [1, 2],
        "temp": [0.5, 0.6],
        "cnt": [100, 200]
    })

    X, y = build_features(df, target="cnt", drop_cols=["instant"])

    assert "instant" not in X.columns
    assert "temp" in X.columns


def test_build_features_returns_correct_shapes():
    df = pd.DataFrame({
        "temp": [0.5, 0.6, 0.7],
        "hum": [0.4, 0.5, 0.6],
        "cnt": [100, 200, 300]
    })

    X, y = build_features(df, target="cnt", drop_cols=[])

    assert X.shape == (3, 2)
    assert y.shape == (3,)