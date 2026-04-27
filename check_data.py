import logging
import yaml
from src.data.load_data import load_raw_data, split_by_year

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

df = load_raw_data(config["data"]["raw_path"])

print("\n--- shape ---")
print(df.shape)

print("\n--- columns ---")
print(df.columns.tolist())

print("\n--- first 3 rows ---")
print(df.head(3))

train_df, live_df = split_by_year(
    df,
    config["data"]["train_year"],
    config["data"]["live_year"]
)

print("\n--- train shape ---")
print(train_df.shape)

print("\n--- live shape ---")
print(live_df.shape)