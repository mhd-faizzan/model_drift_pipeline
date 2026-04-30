# Model Drift Pipeline

An end-to-end MLOps pipeline that detects data drift, retrains the model automatically, and opens a Pull Request with a champion vs new model comparison. No manual intervention required.

Built on the UCI Bike Sharing dataset to simulate a realistic production scenario where a model trained on historical data encounters shifting live traffic patterns over time.

---

## What This Project Does

In production, a model trained today will degrade over time as the real world changes. This pipeline solves that problem by:

1. Serving predictions through a FastAPI endpoint
2. Logging every incoming request to a drift log
3. Running statistical tests to detect when live traffic no longer looks like training data
4. Automatically retraining the model when drift is confirmed
5. Opening a Pull Request with a full metrics comparison so you can review before promoting the new model

---

## How It Works

```
Train model on 2011 data
         ↓
Start prediction API
         ↓
Simulate live requests using 2012 data
         ↓
Run KS test on feature distributions (live vs training)
         ↓
Drift detected in 2+ features?
         ↓ Yes
Backup champion metrics then retrain model
         ↓
Compare new model vs champion
         ↓
Open PR with metrics table and drifted features list
         ↓
Review and merge to promote new model to main
```

---

## Why Drift Is Always Detected in This Project

The model is trained on **2011 data** and the simulation sends **2012 data** to the API. These two years have naturally different seasonal patterns, temperatures, and weather, so the KS test correctly flags drift on every run.

This is intentional. It simulates what happens in a real system where the world changes over time and the model needs to keep up.

**In a real production system**, the drift log would contain actual incoming requests from real users. The pipeline would analyze the last N requests and only trigger a retrain when the live distribution genuinely shifts. The code and pipeline are completely identical. Only the data source changes.

---

## Simulated vs Real Production

| | This Project | Real Production |
|---|---|---|
| Prediction requests | Simulated from 2012 CSV | Real user traffic |
| Drift log | Filled by simulation script | Filled by live API calls |
| Drift trigger | Always (2011 vs 2012 data) | Only when distribution shifts |
| Retrain data | Same 2011 dataset | Rolling window of recent data |
| Pipeline trigger | Manual via GitHub Actions | Can be scheduled or event-driven |

---

## Project Structure

```
model-drift-monitor/
├── src/
│   ├── data/
│   │   ├── load_data.py           # loads and splits raw CSV
│   │   └── simulate_requests.py   # sends live traffic to API
│   ├── features/
│   │   └── build_features.py      # feature engineering
│   └── models/
│       ├── train.py               # trains RandomForest, saves metrics
│       └── detect_drift.py        # KS test drift detection
├── configs/
│   └── config.yaml                # all settings in one place
├── tests/
│   ├── test_drift.py              # drift detection unit tests
│   └── test_features.py           # feature engineering unit tests
├── .github/
│   └── workflows/
│       ├── ci.yml                 # run tests and build Docker image on push
│       └── retrain.yml            # drift detection and retrain pipeline
├── main.py                        # training entry point
├── app.py                         # FastAPI prediction API
├── Dockerfile
└── requirements.txt
```

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Model | scikit-learn RandomForestRegressor |
| API | FastAPI + Uvicorn |
| Drift Detection | SciPy KS test |
| Pipeline | GitHub Actions |
| Containerization | Docker + GitHub Container Registry |
| Data | UCI Bike Sharing Dataset |

---

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for dependency management

### Installation

```bash
git clone https://github.com/mhd-faizzan/model_drift_pipeline.git
cd model_drift_pipeline
python -m venv .venv
source .venv/bin/activate
pip install uv
uv pip install --system -r requirements.txt
```

### Download Dataset

```bash
mkdir -p data/raw
curl -L "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip" -o dataset.zip
unzip dataset.zip -d data/raw
```

### Run Locally

**1. Train the model**
```bash
python main.py
```

**2. Start the API**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

**3. Simulate live traffic**
```bash
python src/data/simulate_requests.py
```

**4. Run drift detection**
```bash
python -c "
import yaml
from src.models.detect_drift import run_drift_detection
with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)
report = run_drift_detection(config)
print(report)
"
```

**5. Check drift report**
```bash
curl http://localhost:8000/drift-report
```

### Run with Docker

```bash
docker build -t model-drift-pipeline .
docker run -p 8000:8000 model-drift-pipeline
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Returns predicted bike count |
| GET | `/drift-report` | Returns latest drift report and total requests logged |

### Example Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "season": 1, "mnth": 1, "hr": 9, "holiday": 0,
    "weekday": 1, "workingday": 1, "weathersit": 1,
    "temp": 0.36, "atemp": 0.3636, "hum": 0.80, "windspeed": 0.0
  }'
```

### Example Response

```json
{
  "predicted_count": 112.35
}
```

---

## Configuration

All settings are in `configs/config.yaml`. Nothing is hardcoded in the code.

```yaml
model:
  n_estimators: 150          # number of trees
  n_jobs: -1                 # use all available CPU cores

drift:
  window_size: 100           # number of recent requests to analyze
  p_value_threshold: 0.05    # KS test significance level
  min_drift_features: 2      # minimum features that must drift to trigger retrain

simulation:
  requests_per_second: 10
  total_requests: 200        # hard stop so simulation exits cleanly
```

---

## GitHub Actions Pipelines

### CI Pipeline

Triggers on every push to `main`.

1. Run unit tests with pytest
2. Build and push Docker image to GitHub Container Registry

### Drift Detection Pipeline

Triggered manually from the Actions tab.

1. Download dataset
2. Train champion model and back up current metrics
3. Start prediction API
4. Simulate live requests
5. Run KS test across all input features
6. If drift detected in 2+ features, retrain model
7. Compare new model vs champion
8. Open PR with metrics table and drifted features list

**To trigger:** Actions > Drift Detection and Retraining > Run workflow > Run workflow

---

## Model Performance

| Metric | Value |
|--------|-------|
| RMSE | 34.9951 |
| MAE | 21.2945 |
| R² | 0.9339 |

Trained on 2011 bike sharing data. 8,645 rows, 11 features, 150 estimators.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## License

MIT