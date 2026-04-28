FROM python:3.11-slim

# prevents .pyc files and enables log streaming
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# copy requirements first — docker caches this layer
# only rebuilds when requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy only what the app needs — not tests, notebooks, data, logs
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/
COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]