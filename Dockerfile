FROM python:3.11-slim

WORKDIR /app

# Requirements copiés EN PREMIER — si inchangés, Docker skip le pip install
COPY api/requirements_api.txt ./req_api.txt
COPY pipeline_ml/requirements_ml.txt ./req_ml.txt
RUN pip install --no-cache-dir -r req_api.txt -r req_ml.txt

# Code source (après pip pour maximiser le layer cache)
COPY api/ ./api/
COPY pipeline_ml/ ./pipeline_ml/
# models/ monté en volume — pas copié dans l'image

EXPOSE 8000
ENV PYTHONPATH=/app

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
