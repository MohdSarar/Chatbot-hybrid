# app/services/pdf_service.py
"""
Service d'extraction de texte depuis un fichier PDF.
"""

import fitz  # PyMuPDF
from fastapi import UploadFile
from app.logging_config import logger

async def extract_text_from_pdf(file: UploadFile) -> str:
    """
    Reçoit un fichier UploadFile, lit son contenu et extrait le texte du PDF.
    Limite le texte extrait à 3000 caractères.
    """
    try:
        contents = await file.read()
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(contents)

        doc = fitz.open(temp_path)
        full_text = "\n".join([page.get_text() for page in doc])
        doc.close()

        logger.info("PDF '%s' traité avec succès.", file.filename)
        return full_text.strip()[:3000]
    except Exception as e:
        logger.error("Erreur lors de la lecture du PDF '%s' : %s", file.filename, str(e))
        return "Erreur lors de la lecture du fichier."
