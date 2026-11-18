FROM python:3.12-slim

LABEL org.opencontainers.image.title="SwimDirector" \
      org.opencontainers.image.description="Watches your swim meet and points the PTZ for you." \
      org.opencontainers.image.source="https://github.com/amurikurthy/SwimDirector"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SWIM_DATA=/data

# libavahi-client3: needed by the NDI runtime bundled in ndi-python
RUN apt-get update \
 && apt-get install -y --no-install-recommends libavahi-client3 libavahi-common3 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app

VOLUME /data
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/state', timeout=4)"
CMD ["uvicorn", "app.web:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
