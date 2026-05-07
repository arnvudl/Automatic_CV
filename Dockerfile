# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Copier les requirements EN PREMIER (layer cache Docker)
# Si les fichiers ne changent pas → pip install est skippé au prochain build
COPY api/requirements_api.txt ./req_api.txt
COPY pipeline_ml/requirements_ml.txt ./req_ml.txt

# BuildKit cache mount : pip réutilise son cache entre les builds
# → packages déjà téléchargés ne sont PAS re-téléchargés
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r req_api.txt -r req_ml.txt

# Code source (copié après pip pour maximiser le cache)
COPY api/ ./api/
COPY pipeline_ml/ ./pipeline_ml/
# models/ est monté en volume — pas copié dans l'image

EXPOSE 8000
ENV PYTHONPATH=/app

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
