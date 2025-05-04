# app/routes/query_routes.py
"""
Route pour interagir avec le chatbot et obtenir une réponse du moteur LLM.
"""
from fastapi import APIRouter, Request, Depends, HTTPException
import gc

from app.services.query_service import (
    SanitizedQueryRequest, 
    process_llm_response,
    format_response,
    handle_query_exception
)
from app.schemas import SessionState, QueryResponse
from app.logging_config import logger

router = APIRouter()

def get_session(request: Request) -> SessionState:
    uid = request.client.host
    if not hasattr(request.app.state, "sessions"):
        request.app.state.sessions = {}
    if uid not in request.app.state.sessions:
        request.app.state.sessions[uid] = SessionState(user_id=uid)
    return request.app.state.sessions[uid]

@router.post("/query", response_model=QueryResponse)
def query_endpoint(req: SanitizedQueryRequest, session: SessionState = Depends(get_session)):
    logger.info(f"Requête reçue: {req.question[:50]}...")

    try:
        response_data = process_llm_response(req.question, req.history, req.profile, session)
        return format_response(response_data, session)
    # Gestion mémoire / erreurs
    except MemoryError:
        logger.critical("ERREUR MÉMOIRE CRITIQUE - Tentative de libération de mémoire")
        gc.collect()
        return QueryResponse(
            reply="Désolé, le service est actuellement surchargé. Veuillez réessayer dans quelques instants.",
            intent="error",
            next_action="retry",
            recommended_course=None
        )
    except Exception as e:
        return handle_query_exception(e)