FROM python:3.12-slim

WORKDIR /app

# v10.45: libgomp1 is required by LightGBM's compiled library (OpenMP
# support) — python:3.12-slim doesn't include it by default. Without this,
# `import lightgbm` fails with "libgomp.so.1: cannot open shared object file".
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .
COPY eod_report.py .
COPY goal_predictor_v1.txt .
COPY goal_predictor_v1_features.json .

CMD ["python", "bot.py"]
