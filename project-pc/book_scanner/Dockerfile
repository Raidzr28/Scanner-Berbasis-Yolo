# Railway build context is this directory (service Root Directory = project-pc/book_scanner).
FROM python:3.12-slim

# opencv-python-headless still links these shared libs at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway injects $PORT; 8080 is just a local default.
ENV PORT=8080
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 180
