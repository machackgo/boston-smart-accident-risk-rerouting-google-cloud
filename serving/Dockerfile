FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# libgomp1 is required by LightGBM for OpenMP multi-threading.
# Without it, LightGBM raises "libgomp.so.1: cannot open shared object file".
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Vertex AI sets AIP_HTTP_PORT; main.py reads it and passes it to uvicorn.
EXPOSE 8080

CMD ["python", "main.py"]
