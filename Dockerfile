FROM python:3.11-slim

WORKDIR /app

# Fusion et installation des dépendances
COPY api/requirements_api.txt ./req_api.txt
COPY pipeline_ml/requirements_ml.txt ./req_ml.txt
RUN pip install --no-cache-dir -r req_api.txt -r req_ml.txt

# Copie des dossiers vitaux
COPY api/ ./api/
COPY pipeline_ml/ ./pipeline_ml/

EXPOSE 8000
ENV PYTHONPATH=/app

# Lancement de FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]