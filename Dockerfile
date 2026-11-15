# EarthScope AI backend (API FastAPI)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EARTHSCOPE_DATA=auto

WORKDIR /app

# dependencias primero para aprovechar la cache de capas
# wheels/ (opcional): paquetes ya bajados con pip download, para no depender de la red
COPY requirements-backend.txt requirements-agent.txt ./
COPY wheels/ /tmp/wheels/
RUN (pip install --no-index --find-links /tmp/wheels -r requirements-agent.txt \
     || pip install --find-links /tmp/wheels -r requirements-agent.txt) \
    && rm -rf /tmp/wheels

# código
COPY api/ api/
COPY science/ science/
COPY export/ export/

# datos que lee la API
COPY data/dataset/ data/dataset/
COPY data/sample/ data/sample/

# usuario sin privilegios
RUN useradd --create-home earthscope \
    && mkdir -p data/.cache \
    && chown -R earthscope:earthscope data/.cache
USER earthscope

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/variables', timeout=4)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
