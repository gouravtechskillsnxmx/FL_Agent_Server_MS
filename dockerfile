# Dockerfile (FastAPI app for Render)
FROM python:3.11-slim

# Working directory
WORKDIR /app

# Install system deps for psycopg2, ffmpeg (audio), build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application source
COPY . /app

# Use unbuffered output for logs
ENV PYTHONUNBUFFERED=1
# Let Render set PORT, but default to 8000
ENV PORT=8000

# Expose the port (Render will override)
EXPOSE ${PORT}

# Use a non-root user (optional but recommended)
RUN groupadd -r appgroup && useradd -r -g appgroup appuser && chown -R appuser:appgroup /app
USER appuser

# Start the app; adjust module:app if your FastAPI app lives elsewhere
CMD ["sh", "-c", "uvicorn agent_server:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 120"]
