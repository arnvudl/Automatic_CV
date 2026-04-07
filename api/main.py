from fastapi import FastAPI, UploadFile, File, Form

app = FastAPI(title="CV API - Test n8n")


@app.post("/api/v1/candidates")
async def receive_cv(
        file: UploadFile = File(...),
        filename: str = Form(...),
        from_email: str = Form(..., alias="from")  # 'from' est un mot réservé en Python
):
    # Lecture basique pour confirmer la réception en log
    content = await file.read()
    size_kb = len(content) / 1024

    print(f"📥 CV reçu : {filename} venant de {from_email} ({size_kb:.2f} KB)")

    # On renvoie le statut attendu par le Switch de n8n
    return {
        "candidate_id": "test_12345",
        "filename": filename,
        "from": from_email,
        "status": "awaiting_review"
    }