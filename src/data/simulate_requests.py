import logging
import time

import pandas as pd
import requests
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000/predict"


def simulate(config: dict) -> None:
    """
    Sends each row of 2012 live data as a POST request to the prediction API.
    Stops after total_requests rows if configured.

    Args:
        config: Full project config dict
    """
    df = pd.read_csv(config["data"]["raw_path"])

    # 2012 data only — yr == 1
    live_df = df[df["yr"] == 1].reset_index(drop=True)

    drop_cols = config["features"]["drop_cols"] + [config["features"]["target"]]
    feature_cols = [col for col in live_df.columns if col not in drop_cols]
    live_df = live_df[feature_cols]

    # cap rows to total_requests if set — prevents simulation running indefinitely
    total_requests = config["simulation"].get("total_requests")
    if total_requests:
        live_df = live_df.head(total_requests)

    logger.info(
        "Starting simulation — %d rows at %d req/s",
        len(live_df),
        config["simulation"]["requests_per_second"]
    )

    sent = 0

    for i, row in live_df.iterrows():
        payload = row.to_dict()

        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            sent += 1

            if sent % 50 == 0:
                logger.info(
                    "Sent %d / %d requests — latest prediction: %s",
                    sent, len(live_df), response.json()
                )

        except requests.exceptions.ConnectionError:
            logger.error("API not reachable at %s — is it running?", API_URL)
            break

        except Exception as e:
            logger.error("Request failed at row %d: %s", i, str(e))
            continue

        time.sleep(1 / config["simulation"]["requests_per_second"])

    logger.info("Simulation complete — %d requests sent", sent)


if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    simulate(config)