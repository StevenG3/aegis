FROM python:3.11-slim
WORKDIR /app
COPY services/screener-service ./service
WORKDIR /app/service
RUN pip install --no-cache-dir -e .
ENV PYTHONPATH=/app/service
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8089"]
