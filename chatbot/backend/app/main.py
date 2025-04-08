# main.py

"""
Exemple d'API FastAPI + pandas, avec chargement de fichiers JSON 
depuis le dossier "content", puis partial matching sur 
l'objectif et les compétences de l'utilisateur.

Endpoints:
 - POST /recommend : reçoit un profil, renvoie une formation adaptée ou un fallback.
 - POST /query : simulation de conversation (réponse fictive).
 - POST /upload-pdf : extrait le contenu d'un fichier PDF.
 - POST /send-email : envoie un e-mail à l'utilisateur, y compris l'historique du chat.

Nécessite: fastapi, uvicorn, pandas, fitz (PyMuPDF), etc.
Pour lancer:
   uvicorn main:app --reload
"""

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import pandas as pd
import json
from pathlib import Path
import fitz  # PyMuPDF
import smtplib
from email.mime.text import MIMEText

from app.logging_config import logger
from dotenv import load_dotenv
import os

load_dotenv()  # charge les variables depuis .env et les place dans os.environ


# --------------------------------------------------
# Initialisation FastAPI
# --------------------------------------------------
app = FastAPI(
    title="Chatbot Formation API (Pandas + dossier content)",
    version="1.0.0",
    description="API de recommandation de formations utilisant pandas, avec chargement JSON depuis 'content'."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Chargement des formations
# --------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parent
DATA_FOLDER = BASE_DIR / "content"
logger.info("Chargement des fichiers depuis : %s", DATA_FOLDER)

def load_formations_to_df(json_dir: Path) -> pd.DataFrame:
    """
    Parcourt tous les fichiers *.json dans json_dir,
    renvoie un DataFrame (titre, objectifs, prerequis, programme, etc.)
    """
    if not json_dir.exists():
        logger.warning("Le dossier %s n'existe pas.", json_dir)
        return pd.DataFrame()

    records = []
    for file in json_dir.glob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                records.append({
                    "titre": data.get("titre", ""),
                    "objectifs": data.get("objectifs", []),
                    "prerequis": data.get("prerequis", []),
                    "programme": data.get("programme", []),
                    "public": data.get("public", []),
                    "lien": data.get("lien", "")
                })
                logger.debug("Fichier chargé : %s", file.name)
        except Exception as e:
            logger.error("Erreur de lecture du fichier %s : %s", file.name, str(e))

    return pd.DataFrame(records)

df_formations = load_formations_to_df(DATA_FOLDER)
logger.info("%d formations chargées", len(df_formations))

# --------------------------------------------------
# Matching logique (mots-clés, scoring)
# --------------------------------------------------
stop_words = {
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "à", "en", 
    "au", "aux", "pour", "avec", "dans", "sur", "par", "se", "son", 
    "sa", "ses", "ce", "cette", "ces", "est", "qui", "que", "dont", 
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles"
}

def extract_keywords(objective: str, knowledge: str) -> List[str]:
    """
    Extrait les tokens significatifs, supprime stop_words
    """
    raw_tokens = (
        objective.lower().replace(",", " ").split() +
        knowledge.lower().replace(",", " ").split()
    )
    tokens = list({t for t in raw_tokens if t and t not in stop_words})
    logger.debug("Mots-clés extraits : %s", tokens)
    return tokens

def partial_match_formations(df: pd.DataFrame, tokens: List[str], niveau_user: str, seuil_score: int) -> pd.DataFrame:
    """
    Filtre et trie les formations par score de matching (tokens + bonus niveau).
    """
    if df.empty or not tokens:
        logger.warning("DF vide ou aucun token fourni")
        return df.iloc[0:0]

    df = df.copy()
    # Concatène objectifs, prerequis, programme en un corpus
    df["corpus"] = df.apply(
        lambda row: " ".join(
            str(x).lower() 
            for lst in [row.get("objectifs", []), row.get("prerequis", []), row.get("programme", [])]
            for x in (lst if isinstance(lst, list) else [lst])
        ),
        axis=1
    )

    def compute_score(row):
        text = row["corpus"]
        # 1 point par token trouvé
        score = sum(1 for t in tokens if t in text)
        niveau_formation = row.get("niveau", "").lower()
        prerequis = row.get("prerequis", [])

        # Bonus si user=deb et formation=deb ou prerequis vide
        if niveau_user == "débutant" and ("débutant" in niveau_formation or not prerequis):
            score += 2
        # Bonus si user=avancé et formation=avancé
        elif niveau_user == "avancé" and "avancé" in niveau_formation:
            score += 1
        return score

    df["score"] = df.apply(compute_score, axis=1)
    logger.info(
        "Top formations (tri par score) :\n%s",
        df[["titre", "score"]].sort_values(by="score", ascending=False).to_string(index=False)
    )
    return df[df["score"] >= seuil_score].sort_values(by="score", ascending=False)

# --------------------------------------------------
# Modèles Pydantic
# --------------------------------------------------
class UserProfile(BaseModel):
    name: str
    email: Optional[EmailStr] = None
    objective: str
    level: str
    knowledge: str
    pdf_content: Optional[str] = None
    recommended_course: Optional[str] = None

# Ajout d'un modèle ChatMessage
class ChatMessage(BaseModel):
    role: str  # 'user' ou 'assistant'
    content: str

class SendEmailRequest(BaseModel):
    profile: UserProfile
    chatHistory: List[ChatMessage] = []

class RecommendRequest(BaseModel):
    profile: UserProfile

class RecommendResponse(BaseModel):
    recommended_course: str
    reply: str
    details: Optional[dict] = None

class QueryRequest(BaseModel):
    profile: UserProfile
    history: List[dict] = []
    question: str

class QueryResponse(BaseModel):
    reply: str
    

# --------------------------------------------------
# Envoi email via Gmail (App password)
# --------------------------------------------------
def send_email_notification(to: str, subject: str, body: str):
    """
    Envoie un email en texte brut via Gmail SMTP (port 587).
    Nécessite les variables d'env GMAIL_USER / GMAIL_APP_PASS
    et la validation 2FA activée côté Google.
    """

    gmail_user = os.environ.get("GMAIL_USER")
    gmail_app_password = os.environ.get("GMAIL_APP_PASS")


    message = MIMEText(body, "plain", "utf-8")
    message["From"] = gmail_user
    message["To"] = to
    message["Subject"] = subject

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            # Optionnel: server.set_debuglevel(True)  # pour + de logs
            server.ehlo()
            server.starttls()
            server.ehlo()

            # Se connecte en utilisant l'app password
            server.login(gmail_user, gmail_app_password)

            server.sendmail(gmail_user, [to], message.as_string())

        print(f"[INFO] Email envoyé à {to}")
    except Exception as e:
        print(f"[ERROR] Erreur envoi email à {to} : {e}")

# --------------------------------------------------
# Construire le corps d'email intégrant l'historique
# --------------------------------------------------

def build_email_body(profile: UserProfile, chat_history: List[ChatMessage]) -> str:
    lines = []
    lines.append(f"Bonjour {profile.name},\n")
    lines.append(f"Objectif : {profile.objective}")
    lines.append(f"Niveau : {profile.level}")
    lines.append(f"Compétences : {profile.knowledge}")
    lines.append(f"Formation recommandée : {profile.recommended_course or 'Aucune'}\n")

    lines.append("=== Historique de Chat ===")

    for msg in chat_history:
        role_label = "USER" if msg.role == "user" else "ASSISTANT"

        # Tente de parser le JSON
        if msg.role == "assistant":
            try:
                data = json.loads(msg.content)
                # data = { "reply": "...", "course": "...", "details": {...} }
                # On reconstruit un affichage plus sympa :
                lines.append(f"{role_label}: {data['reply']}")  # ex "Voici une formation..."

                # Si tu veux afficher le 'course' comme "Formation Intégration ..."
                if 'course' in data:
                    lines.append(f"  -> Formation : {data['course']}")

                # Idem pour details, si c'est un dict
                if 'details' in data and isinstance(data['details'], dict):
                    # On peut extraire objectifs, prerequis, programme, ...
                    obj = data['details'].get('objectifs', [])
                    pre = data['details'].get('prerequis', [])
                    prog = data['details'].get('programme', [])

                    # ex: lines.append(f"    Objectifs: {', '.join(obj)}")

            except json.JSONDecodeError:
                # Si pas du JSON => on affiche tel quel
                lines.append(f"{role_label}: {msg.content}")
        else:
            # C'est un message user
            lines.append(f"{role_label}: {msg.content}")

    lines.append("\nMerci de votre visite.")
    return "\n".join(lines)


# --------------------------------------------------
# Endpoint /query (réponse fictive)
# --------------------------------------------------
@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    logger.info("/query appelé avec question : %s", req.question)
    return QueryResponse(reply=f"Réponse fictive à '{req.question.strip()}'. (Pas de LLM)")

# --------------------------------------------------
# Endpoint /recommend
# --------------------------------------------------
@app.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(r: RecommendRequest):
    """
    Reçoit un UserProfile, calcule la formation la plus adaptée, 
    ou renvoie un fallback si rien ne correspond.
    """
    profile = r.profile
    logger.info("/recommend appelé pour utilisateur : %s", profile.name)

    # Extraction de mots-clés
    tokens = extract_keywords(profile.objective, profile.knowledge + (profile.pdf_content or ""))
    niveau = profile.level.lower().strip()
    seuil = 2

    # Matching
    matched_df = partial_match_formations(df_formations, tokens, niveau_user=niveau, seuil_score=seuil)
    if not matched_df.empty:
        match = matched_df.iloc[0]
        logger.info("Formation recommandée : %s", match["titre"])
        return RecommendResponse(
            recommended_course=match["titre"],
            reply="Voici une formation qui correspond à votre profil.",
            details={
                "objectifs": match["objectifs"],
                "prerequis": match["prerequis"],
                "programme": match["programme"],
                "lien": match["lien"]
            }
        )
    else:
        logger.warning("Aucune formation ne correspond au profil fourni")
        return RecommendResponse(
            recommended_course="Aucune formation pertinente",
            reply="Aucune formation ne correspond aux mots-clés fournis.",
            details=None
        )

# --------------------------------------------------
# Endpoint /upload-pdf
# --------------------------------------------------
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Reçoit un PDF, en extrait le texte via PyMuPDF, 
    et le renvoie tronqué à 3000 caractères max.
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
        return {"content": full_text.strip()[:3000]}
    except Exception as e:
        logger.error("Erreur lors de la lecture du PDF '%s' : %s", file.filename, str(e))
        return {"content": "Erreur lors de la lecture du fichier."}


# --------------------------------------------------
# Endpoint /send-email
# --------------------------------------------------
@app.post("/send-email")
def send_email(req: SendEmailRequest, background_tasks: BackgroundTasks):
    """
    Reçoit :
      - profile (UserProfile)
      - chatHistory (liste de ChatMessage)
    Construit un email récapitulatif et l'envoie, si profile.email est renseigné.
    """
    profile = req.profile
    history = req.chatHistory

    if not profile.email:
        return {"status": "Aucune adresse email fournie"}

    subject = "Votre récapitulatif de session Chatbot"
    # Construit un corps de mail intégrant l'historique
    body = build_email_body(profile, history)

    background_tasks.add_task(
        send_email_notification,
        profile.email,
        subject,
        body
    )
    return {"status": "Email en cours d'envoi"}
