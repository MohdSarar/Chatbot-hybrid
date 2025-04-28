# app/routes/upload_routes.py
"""
Route pour uploader un fichier PDF et extraire son contenu texte.
"""

from fastapi import APIRouter, UploadFile, File
from app.services.pdf_service import extract_text_from_pdf
from app.logging_config import logger

router = APIRouter()

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint pour recevoir un fichier PDF et renvoyer son contenu texte (limité à 3000 caractères).
    """
    print(f"[INFO] Fichier PDF reçu pour traitement : {file.filename}")
    logger.info("Fichier PDF reçu pour traitement : %s", file.filename)

    return {"content": await extract_text_from_pdf(file)}
