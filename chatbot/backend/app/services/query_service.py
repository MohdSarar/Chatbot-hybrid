# app/services/query_service.py
"""
Service pour traiter les requêtes de l'utilisateur et interagir avec le moteur LLM.
Version mise à jour pour utiliser RNCPRetrievalService au lieu de LangChainRAGService.
"""
from __future__ import annotations
from fastapi import HTTPException
from pydantic import BaseModel, validator
import gc
import json
from pathlib import Path

from app.schemas import UserProfile, SessionState, QueryResponse
from app.logging_config import logger
from app.utils import search_and_format_courses
import app.globals as globs
from app.formation_search import FormationSearch as fs

import logging, json
from typing import List, Dict


# Constantes
DATA_FOLDER = Path(__file__).resolve().parent.parent / "content"

class SanitizedQueryRequest(BaseModel):
    """Requête étendue avec validation des entrées."""
    profile: UserProfile
    history: list = []
    question: str

def get_llm_engine():
    if globs.llm_engine is None:
        raise HTTPException(503, "Service en cours d'initialisation")
    return globs.llm_engine


def external_warning(title: str) -> str:
    return (
        f" {title} ne sont pas commercialisées par Beyond Expertise. "
        "Les informations suivantes sont fournies uniquement à titre d’orientation :"
    )


logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# 1.  Helpers réutilisables
# ──────────────────────────────────────────────────────────────
def _truncate(lst: list, n: int) -> list:
    return lst[-n:] if len(lst) > n else lst


# ──────────────────────────────────────────────────────────────
# 2.  Fonction principale simplifiée
# ──────────────────────────────────────────────────────────────

def process_llm_response(
    question: str,
    history: List[dict],
    profile: UserProfile,
    session: SessionState,
    last_recommended_course: dict | None = None
) -> Dict:
    logger.info("Process question: %.50s", question)

    # 0) Si la requête client fournit une précédente recommended_course, on la stocke
    if last_recommended_course:
        session.recommended_course = last_recommended_course
    # Récupération du moteur LLM
    try:
        engine = get_llm_engine()
    except Exception as exc:
        logger.error("Init engine KO: %s", exc)
        return _error("init_error")

    # Ne conserver que les 20 derniers messages
    history = _truncate(history, 20)

    # 1) Détection d’intention
    intent = engine.detect_intent(question, history)
    logger.info("Intent = %s", intent)

    # 2) Recherche des formations pertinentes
    formatted_text, course_source = search_and_format_courses(question, k=5)
    print(f"\n\n Course Source : \n{course_source}\n\n")
    print(f"\n\n Formatted Text : \n#{str(formatted_text)}#\n\n")

    # Si on n’a rien trouvé ET que l’utilisateur pose une question sur la formation recommandée
    if not formatted_text and session.recommended_course:
        rc = session.recommended_course
        # Extraction selon le nouveau schéma français
        titre       = rc.get('titre', 'N/A')
        objectifs   = rc.get('objectifs', [])
        public_cible= rc.get('public_cible', [])
        duree       = rc.get('duree', 'N/A')
        tarif       = rc.get('tarif', 'N/A')
        lieu        = rc.get('lieu', 'N/A')
        certifiant  = rc.get('certifiant', False)
        source      = rc.get('_source', 'internal')

        # Re-création du bloc pour le prompt
        formatted_text = (
            "Voici la formation précédemment recommandée :\n"
            f"- Titre           : {titre}\n"
            f"- Objectifs       : {', '.join(objectifs) if objectifs else 'N/A'}\n"
            f"- Public ciblé    : {', '.join(public_cible) if public_cible else 'N/A'}\n"
            f"- Durée           : {duree}\n"
            f"- Tarif           : {tarif}\n"
            f"- Leiu            : {lieu}\n"
            f"- Certifiante     : {'Oui' if certifiant else 'Non'}\n"
            f"- _source         : {source}\n"
        )
        course_source = source

    # 3) Construction du prompt « prêt »
    if course_source == "external":
        # Formations externes RNCP : on ajoute un warning et une consigne RNCP claire
        base_prompt = (
            external_warning("ces formations") + "\n\n"
            f"{formatted_text.rstrip()}\n\n"
            "À partir des formations du RNCP listées ci-dessous, choisissez une seule formation la plus pertinente "
            "pour l'utilisateur et indiquez-la dans le champ `recommended_course` du JSON.\n\n"
            f"Question utilisateur : {question}\n"
        )
    else:
        # Cas interne : prompt standard
        base_prompt = ""
        if formatted_text:
            base_prompt += formatted_text.rstrip() + "\n\n"
            if intent == "recherche_filtrée":
                criteria = extract_criteria_from_question(question)
                intent_instr = build_intent_instruction(intent, criteria)

        intent_instr = build_intent_instruction(intent).strip()


        if intent_instr:
            base_prompt += intent_instr + "\n\n"
        base_prompt += f"Question utilisateur : {question}\n"


    print(f"\n\n Base Prompt : \n\n {base_prompt}\n\n\n")
    # 4) Appel au LLMEngine en mode « ready prompt » (injection historique +system JSON spec)
    response = engine.generate_response(
        base_prompt,
        chat_history=history,
        profile=profile,
        session=session,
        is_ready_prompt=True,
        course_source=course_source,
    )
    
    rc = response.get("recommended_course")
    if isinstance(rc, dict):
        session.recommended_course = rc
    elif session.recommended_course:
        response["recommended_course"] = session.recommended_course


    return response




# ──────────────────────────────────────────────────────────────
# 3.  Réponses d’erreur homogènes
# ──────────────────────────────────────────────────────────────
def _error(code: str) -> Dict:
    msgs = {
        "init_error":  "Désolé, un problème interne empêche le traitement de votre demande.",
        "no_course":   "Je n’ai pas trouvé de formation correspondante. Pouvez-vous préciser ?",
    }
    return {"answer": msgs.get(code, msgs["init_error"]),
            "recommended_course": None,
            "next_action": "error",
            "intent": "error"}


def format_response(response_data: dict, session: SessionState) -> QueryResponse:
    """Formate les données de réponse en QueryResponse et met à jour la session."""
    
    if response_data is None:
        return QueryResponse(
            reply="Désolé, une erreur est survenue lors du traitement de votre requête.",
            intent="error",
            next_action="retry",
            recommended_course=None
        )

    # 1) On récupère la recommended_course et l'intent renvoyés par le LLM
    rc = response_data.get("recommended_course")
    intent = response_data.get("intent", "")

    # 2) Mise à jour de la session uniquement lors de la recommandation initiale
    if intent == "recommandation" and isinstance(rc, dict):
        session.recommended_course = rc
    # 3) Si le LLM répond `recommended_course: null` et qu'on a une valeur en session,
    #    on la remet dans response_data pour ne pas la perdre
    elif rc is None and session.recommended_course:
        response_data["recommended_course"] = session.recommended_course

    # 4) Construction et retour du QueryResponse final
    return QueryResponse(
        reply=response_data.get("answer", ""),
        intent=response_data.get("intent", "fallback"),
        next_action=response_data.get("next_action", "follow_up"),
        recommended_course=response_data.get("recommended_course")
    )



def handle_query_exception(e: Exception) -> QueryResponse:
    """Gestion des exceptions pour l'endpoint query."""
    logger.error(f"Erreur non gérée dans query_endpoint: {str(e)}", exc_info=True)
    return QueryResponse(
        reply="Une erreur est survenue lors du traitement de votre demande. Notre équipe technique a été notifiée.",
        intent="error",
        next_action="contact_support",
        recommended_course=None
    )

def build_intent_instruction(
    intent: str,
    criteria: dict | None = None
) -> str:
    """
    Génère une instruction textuelle à injecter dans le prompt LLM selon l'intention utilisateur.
    """

    match intent:
        case "recommandation":
            return (
                "\nÀ partir des résultats proposés, identifiez une formation unique à recommander à l'utilisateur. "
                "Présentez-la de manière claire et engageante dans la réponse (`answer`) "
                "ne mentionnez pas les infos secondaires dans la réponse."
                "et indiquez-la dans le champ `recommended_course` du JSON."
            )
        case "liste_internes":
            internes = globs.formation_search.filter_formations(_source="internal")
            if not internes:
                return "\nAucune formation interne n'est disponible pour le moment."

            # 2) On construit le préfixe listant les formations internes
            prefix = "Voici les formations proposées par Beyond Expertise :\n"
            for f in internes:
                titre = f.get("titre", "–")
                duree = f.get("duree", "N/A")
                tarif = f.get("tarif", "N/A")
                prefix += f"- {titre} (Durée : {duree}, Tarif : {tarif})\n"
            prefix += "\n"

            # 3) On garde votre instruction d’origine
            return (
                prefix +
                "appuyez vous sur la liste ci-dessus pour répondre de manière claire et professionnelle à la question de l'utilisateur."
            )
        case "liste_externes":
            return (
                "\nPrésentez une liste example de 10 formations certifiantes du RNCP de manière claire, en précisant qu'elles ne sont pas commercialisées par Beyond Expertise."
            )
        case "info_tarif":
            return (
                "\nFournissez CLAIREMENT les informations sur les tarifs des formations mentionnées dans les résultats ou dans la formation précédemment recommandée. "
                "Assurez-vous que votre réponse dans le champ 'answer' contient explicitement le tarif. "
                "Conservez **toute** la formation recommandée précédente dans le champ `recommended_course` du JSON, "
                "et remplissez-y le champ `tarif` avec la valeur extraite du bloc ci-dessus."
            )
        case "info_duree":
            return "\nExpliquez la durée de la ou les formations en question."
        case "info_certification":
            return "\nIndiquez si les formations proposées sont certifiantes ou non."
        case "comparaison":
            return "\nComparez les deux formations les plus pertinentes dans les résultats selon leurs caractéristiques clés."
        case "recherche_filtrée":                 
            prefix = ""
            if criteria:
                filtered = globs.formation_search.filter_formations(**criteria)
                if filtered:
                    prefix = "Voici les formations correspondant à vos critères :\n"
                    for f in filtered:
                        titre = f.get("titre", "–")
                        duree = f.get("duree", "N/A")
                        tarif = f.get("tarif", "N/A")
                        prefix += f"- {titre} (Durée : {duree}, Tarif : {tarif})\n"
                    prefix += "\n"
            # 2) Puis on ajoute l’instruction classique
            return (
                prefix +
                "Si la question contient un ou plusieurs filtres (ex. : certifiante, à distance), "
                "répondez en listant **uniquement** les formations qui correspondent à ces critères, "
                "en précisant leurs caractéristiques principales."
            )
        case "fallback":
            return (
                "\nFormulez une réponse utile et informative à la question même si elle ne correspond pas à une intention précise et même si elle est pas pertinente avec les informations mentionnées au dessus."
            )
        case _:
            return ""





import re

def extract_criteria_from_question(question: str) -> dict:
    """
    Analyse la question pour repérer des filtres :
    - certifiant, modalité, durée…
    - ET des critères de prix (gratuit, moins de X€, entre X et Y€, plus de X€)
    Retourne un dict utilisable par filter_formations.
    """
    q = question.lower()
    criteria = {}

    # -- filtres existants --
    if "certifiante" in q or "certifiantes" in q:
        criteria["certifiant"] = True
    if "à distance" in q or "en ligne" in q:
        criteria["modalite"] = "à distance"
    elif "sur site" in q or "présentiel" in q:
        criteria["modalite"] = "sur site"

    # -- critère prix : gratuit --
    if "gratuit" in q:
        criteria["tarif_max"] = 0.0

    # -- critère prix : entre X et Y --
    m = re.search(r'entre\s*(\d+[\d\s]*)\s*(?:€|eur)\s*(?:et|-)\s*(\d+[\d\s]*)', q)
    if m:
        low = float(m.group(1).replace(" ", ""))
        high = float(m.group(2).replace(" ", ""))
        criteria["tarif_min"] = low
        criteria["tarif_max"] = high

    # -- critère prix : moins de X --
    m = re.search(r'moins de\s*(\d+[\d\s]*)\s*(?:€|eur)', q)
    if m:
        criteria["tarif_max"] = float(m.group(1).replace(" ", ""))

    # -- critère prix : plus de X ou à partir de X --
    m = re.search(r'(?:plus de|à partir de)\s*(\d+[\d\s]*)\s*(?:€|eur)', q)
    if m:
        criteria["tarif_min"] = float(m.group(1).replace(" ", ""))

    return criteria
