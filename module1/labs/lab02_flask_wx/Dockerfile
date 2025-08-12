FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
ENV PORT=8080
EXPOSE 8080
CMD ["gunicorn","-b","0.0.0.0:${PORT}","app:app"]
