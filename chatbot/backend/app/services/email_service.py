# app/services/email_service.py
"""
Service d'envoi d'email et de construction du contenu du mail.
"""

import os
import smtplib
import json
from email.mime.text import MIMEText
from typing import List
from app.schemas import UserProfile, ChatMessage
from app.logging_config import logger

def send_email_notification(to: str, subject: str, body: str):
    """
    Envoie un email en texte brut via SMTP Gmail.
    Nécessite GMAIL_USER et GMAIL_APP_PASS dans les variables d'environnement.
    """
    gmail_user = os.environ.get("GMAIL_USER")
    gmail_app_password = os.environ.get("GMAIL_APP_PASS")

    message = MIMEText(body, "plain", "utf-8")
    message["From"] = gmail_user
    message["To"] = to
    message["Subject"] = subject

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(gmail_user, gmail_app_password)
            server.sendmail(gmail_user, [to], message.as_string())

        logger.info("Email envoyé avec succès à %s", to)
    except Exception as e:
        logger.error("Erreur envoi email à %s : %s", to, str(e))

def build_email_body(profile: UserProfile, chat_history: List[ChatMessage]) -> str:
    """
    Construit le corps de l'email récapitulatif de la session utilisateur.
    """
    lines = []
    lines.append(f"Bonjour {profile.name},\n")
    lines.append(f"Objectif : {profile.objective}")
    lines.append(f"Niveau : {profile.level}")
    lines.append(f"Compétences : {profile.knowledge}")
    lines.append(f"Formation recommandée : {profile.recommended_course or 'Aucune'}\n")
    lines.append("=== Historique de Chat ===")

    for msg in chat_history:
        role_label = "USER" if msg.role == "user" else "ASSISTANT"
        if msg.role == "assistant":
            try:
                data = json.loads(msg.content)
                lines.append(f"{role_label}: {data['reply']}")
                if 'course' in data:
                    lines.append(f"  -> Formation : {data['course']}")
            except json.JSONDecodeError:
                lines.append(f"{role_label}: {msg.content}")
        else:
            lines.append(f"{role_label}: {msg.content}")

    lines.append("\nMerci de votre visite.")
    return "\n".join(lines)
