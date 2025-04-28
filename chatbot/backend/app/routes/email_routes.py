# app/routes/email_routes.py
"""
Route pour envoyer un email récapitulatif à l'utilisateur après une session.
"""

from fastapi import APIRouter, BackgroundTasks
from app.schemas import SendEmailRequest
from app.services.email_service import send_email_notification, build_email_body
from app.logging_config import logger

router = APIRouter()

@router.post("/send-email")
def send_email(req: SendEmailRequest, background_tasks: BackgroundTasks):
    """
    Endpoint pour envoyer un email contenant l'historique du chat et les détails de la session utilisateur.
    """
    profile = req.profile
    history = req.chatHistory

    if not profile.email:
        logger.warning("Aucune adresse email fournie pour l'envoi.")
        print("[WARNING] Aucun email fourni dans la requête /send-email.")
        return {"status": "Aucune adresse email fournie"}

    subject = "Votre récapitulatif de session Chatbot"
    body = build_email_body(profile, history)

    background_tasks.add_task(
        send_email_notification,
        profile.email,
        subject,
        body
    )
    print(f"[INFO] Envoi de l'email en arrière-plan vers : {profile.email}")
    logger.info("Email en cours d'envoi vers : %s", profile.email)

    return {"status": "Email en cours d'envoi"}
