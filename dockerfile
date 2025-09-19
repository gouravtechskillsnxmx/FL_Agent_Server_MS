FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lis# Dockerfile (for FastAPI on Render)
FROM python:3.11-slim

# set a working directory
WORKDIR /app

# system deps needed for psycopg2, ffmpeg (optional if pydub requires), and building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy application source
COPY . /app

# ensure environment variable PORT used by Render
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# create an unprivileged user (optional)
RUN adduser --disabled-password --gecos "" appuser || true
USER appuser

# expose port (Render will override)
EXPOSE ${PORT}

# start uvicorn (use env var PORT)
CMD ["sh", "-c", "uvicorn agent_server:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 120"]
ts/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8000
CMD ["uvicorn", "agent_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
