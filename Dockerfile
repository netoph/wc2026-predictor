FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/tracker data/odds

EXPOSE 8026

# Start with 0 workers timeout for model training at startup
CMD ["uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "8026", "--timeout-keep-alive", "120"]
