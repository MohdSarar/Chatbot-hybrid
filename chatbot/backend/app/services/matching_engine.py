# app/services/matching_engine.py
"""
Service de matching de formations basé sur des mots-clés et le niveau utilisateur.
"""

import pandas as pd
from typing import List
from app.logging_config import logger

stop_words = {
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "à", "en", 
    "au", "aux", "pour", "avec", "dans", "sur", "par", "se", "son", 
    "sa", "ses", "ce", "cette", "ces", "est", "qui", "que", "dont", 
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles"
}

def extract_keywords(objective: str, knowledge: str) -> List[str]:
    """
    Extrait les tokens significatifs en supprimant les mots inutiles (stop words).
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
        score = sum(1 for t in tokens if t in text)
        niveau_formation = row.get("niveau", "").lower()
        prerequis = row.get("prerequis", [])

        if niveau_user == "débutant" and ("débutant" in niveau_formation or not prerequis):
            score += 2
        elif niveau_user == "avancé" and "avancé" in niveau_formation:
            score += 1
        return score

    df["score"] = df.apply(compute_score, axis=1)
    logger.info(
        "Top formations (tri par score) :\n%s",
        df[["titre", "score"]].sort_values(by="score", ascending=False).to_string(index=False)
    )
    return df[df["score"] >= seuil_score].sort_values(by="score", ascending=False)

def custom_recommendation_scoring(profile, df: pd.DataFrame) -> pd.DataFrame:
    """
    Évalue la compatibilité entre le profil utilisateur et les formations.
    """
    if df.empty:
        return df

    tokens_objectif = extract_keywords(profile.objective, "")
    tokens_knowledge = extract_keywords("", profile.knowledge)

    df = df.copy()

    def score_row(row):
        objectifs = " ".join(row.get("objectifs", [])).lower()
        prerequis = " ".join(row.get("prerequis", [])).lower()
        programme = " ".join(row.get("programme", [])).lower()

        score = 0
        score += sum(1 for t in tokens_objectif if t in objectifs or t in programme)
        score += sum(1 for t in tokens_knowledge if t in prerequis)

        niveau = row.get("niveau", "").lower()
        if profile.level.lower() == niveau:
            score += 1

        return score

    df["score"] = df.apply(score_row, axis=1)
    return df[df["score"] > 0].sort_values(by="score", ascending=False)
