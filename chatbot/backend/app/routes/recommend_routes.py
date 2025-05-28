# app/routes/recommend_routes.py
"""
Route pour recommander une formation adaptée à l'utilisateur.
"""

from fastapi import APIRouter
from app.schemas import RecommendRequest, RecommendResponse
from app.services.matching_engine import custom_recommendation_scoring
from app.services.data_loader import load_formations_to_df
from pathlib import Path
from app.logging_config import logger
import app.globals as globs

router = APIRouter()

DATA_FOLDER = Path(__file__).resolve().parent.parent / "content"
df_formations = load_formations_to_df(DATA_FOLDER)

@router.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(r: RecommendRequest):
    profile = r.profile
    logger.info("Réception d'une requête /recommend pour l'utilisateur : %s", profile.name)
    matched_df = custom_recommendation_scoring(profile, df_formations)
    # Mets à jour le profil utilisateur
    globs.llm_counselor.set_user_profile_from_pydantic(profile)
    
    # ✅ DEBUG: Log profile after setting
    print(f"🔍 COUNSELOR CONTEXT AFTER:")
    print(f"   Nom: {globs.llm_counselor.ctx.nom}")
    print(f"   Objectif: {globs.llm_counselor.ctx.objectif}")
    print(f"   Compétences: {globs.llm_counselor.ctx.competences}")
    print("=====================================\n")
    # Mets à jour l'historique de conversation
    globs.llm_counselor._init_conversation_history()

    if not matched_df.empty:
        best_match = matched_df.iloc[0]
        titre = best_match["titre"]
        logger.info("Formation recommandée déterminée : %s", titre)
        return RecommendResponse(
            recommended_course=titre,
            reply="Voici une formation qui correspond à votre profil et à vos objectifs :",
            details={
                "objectifs": best_match["objectifs"],
                "prerequis": best_match["prerequis"],
                "programme": best_match["programme"],
                "lien": best_match["lien"]
            }
        )
    else:
        fallback = df_formations[df_formations["prerequis"].apply(lambda x: not x or len(x) == 0)]
        if not fallback.empty:
            choice = fallback.sample(1).iloc[0]
            titre = choice["titre"]
            logger.info("Aucune formation idéale trouvée, fallback sur : %s", titre)
            return RecommendResponse(
                recommended_course=titre,
                reply="Aucune formation ne correspond exactement à votre profil, mais voici une formation accessible sans prérequis :",
                details={
                    "objectifs": choice["objectifs"],
                    "prerequis": choice["prerequis"],
                    "programme": choice["programme"],
                    "lien": choice["lien"]
                }
            )
        else:
            return RecommendResponse(
                recommended_course="",
                reply="Désolé, aucune recommandation de formation n'a pu être déterminée pour votre profil.",
                details={}
            )
