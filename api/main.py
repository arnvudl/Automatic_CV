from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cv_api")

app = FastAPI(title="Automatic CV API", version="0.1.0")

# Sécurisation CORS : On restreint l'accès à ton domaine n8n
# En production, ne jamais laisser ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://n8n.lony.app"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "online", "message": "API prêt pour n8n"}


@app.post("/api/v1/candidates")
async def receive_cv(
        file: UploadFile = File(...),
        filename: str = Form(...),
        from_email: str = Form(..., alias="from")
):
    try:
        content = await file.read()

        logger.info(f"✅ CV REÇU : {filename}")
        logger.info(f"📧 EXPÉDITEUR : {from_email}")

        # Simulation du succès pour le test de workflow
        return {
            "status": "awaiting_review",
            "candidate_id": "cand_dev_test",
            "filename": filename,
            "message": "Intégration réussie."
        }

    except Exception as e:
        logger.error(f"❌ Erreur : {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur interne")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)