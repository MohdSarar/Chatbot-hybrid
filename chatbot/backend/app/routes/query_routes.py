# app/routes/query_routes.py
"""
Route pour interagir avec le chatbot et obtenir une réponse du moteur LLM.
"""

from fastapi import APIRouter, Depends, Request
from app.schemas import QueryRequest, QueryResponse, SessionState, UserProfile
from app.llm_engine import LLMEngine
from app.services.data_loader import load_formations_to_df
from pathlib import Path
from app.logging_config import logger

router = APIRouter()

DATA_FOLDER = Path(__file__).resolve().parent.parent / "content"
df_formations = load_formations_to_df(DATA_FOLDER)

llm_engine = LLMEngine(df_formations)
print("[INFO] Moteur LLM initialisé avec succès.")
logger.info("Moteur LLM initialisé avec succès.")

def get_session(request: Request) -> SessionState:
    """
    Gestion ultra simple de la session utilisateur basée sur l'adresse IP.
    """
    uid = request.client.host
    if not hasattr(request.app.state, "sessions"):
        request.app.state.sessions = {}
    return request.app.state.sessions.setdefault(uid, SessionState(user_id=uid))

def process_llm_response(question: str, history: list, profile: UserProfile, session: SessionState) -> str:
    """
    Génère une réponse en fonction de l'historique du chat et du profil utilisateur.
    """
    chat_history = []
    for i in range(0, len(history), 2):
        if i + 1 < len(history):
            user_msg = history[i]
            assistant_msg = history[i + 1]
            if user_msg.role == "user" and assistant_msg.role == "assistant":
                chat_history.append((user_msg.content, assistant_msg.content))
    try:
        llm_result = llm_engine.generate_response(question=question, chat_history=chat_history, profile=profile, session=session)
        return llm_result.get("answer", "")
    except Exception as e:
        logger.error("Erreur lors de la génération de réponse LLM : %s", str(e))
        print(f"[ERROR] Erreur génération réponse LLM : {e}")
        return "Désolé, une erreur est survenue lors de la génération de la réponse."

@router.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest, session: SessionState = Depends(get_session)):
    """
    Endpoint pour soumettre une question au moteur LLM via FastAPI.
    """
    logger.info("Réception d'une requête /query : %s", req.question)
    print(f"[INFO] Requête reçue sur /query avec question : {req.question}")
    reply_text = process_llm_response(req.question, req.history, req.profile, session)
    return QueryResponse(reply=reply_text)
