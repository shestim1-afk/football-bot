FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .
COPY eod_report.py .
COPY goal_predictor_v1.txt .
COPY goal_predictor_v1_features.json .

CMD ["python", "bot.py"]
