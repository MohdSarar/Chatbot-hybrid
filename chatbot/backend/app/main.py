"""
main.py

Exemple d'API FastAPI + pandas, avec chargement de fichiers JSON 
depuis le dossier "content", puis partial matching sur 
l'objectif et les compétences de l'utilisateur.

- POST /recommend : reçoit un profil, renvoie une formation adaptée ou un fallback.
- POST /query : simulation de conversation (réponse fictive).

Nécessite: fastapi, uvicorn, pandas, etc.
Pour lancer:
    uvicorn main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

import pandas as pd
import json
from pathlib import Path

# ===========================================
# == Création de l'API FastAPI            ==
# ===========================================
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

# ===========================================
# == Détermination du chemin absolu         ==
# ===========================================
# L'objectif est de pointer vers le dossier "content" 
# où se trouvent désormais tous les fichiers .json.

CURRENT_FILE = Path(__file__).resolve()
print("[DEBUG] __file__ =", __file__)
print("[DEBUG] CURRENT_FILE =", CURRENT_FILE)

# Supposons que main.py est dans 'app/' (ou la racine).
# .parent => dossier contenant main.py
# .parent.parent => si besoin de remonter un niveau supplémentaire
# Ajuster selon l'emplacement réel :
BASE_DIR = CURRENT_FILE.parent  # si main.py est à la racine, c'est suffisant
# Si main.py est dans 'app/', alors => BASE_DIR = CURRENT_FILE.parent.parent
# À adapter selon la hiérarchie exacte.

# print("[DEBUG] BASE_DIR =", BASE_DIR)

DATA_FOLDER = BASE_DIR / "content"
# print("[DEBUG] DATA_FOLDER =", DATA_FOLDER, "| exists?", DATA_FOLDER.exists())

# ============================================
# == Chargement des formations en DataFrame ==
# ============================================

def load_formations_to_df(json_dir: Path) -> pd.DataFrame:
    """
    Parcourt tous les fichiers *.json dans 'json_dir',
    et construit un DataFrame avec:
      - titre
      - objectifs (liste)
      - prerequis (liste)
      - programme (liste)
      - public (liste)
      - lien
    Ajoute des prints pour vérifier le contenu.
    """
    # print("\n[DEBUG] Chargement des formations depuis:", json_dir)
    if not json_dir.exists():
        # print("[WARNING] Le dossier n'existe pas. Aucun fichier JSON ne sera chargé.")
        return pd.DataFrame()

    records = []
    nb_files = 0

    for file in json_dir.glob("*.json"):
        nb_files += 1
        print("[DEBUG] Ouverture du fichier:", file)
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
        # print("[DEBUG] Fichier chargé avec succès:", file.name)

    if nb_files == 0:
        # print("[WARNING] Aucun fichier .json trouvé dans le dossier.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # print(f"[DEBUG] Nombre de formations chargées: {len(df)}")
    return df

# Chargement effectif
df_formations = load_formations_to_df(DATA_FOLDER)
# print("\n[DEBUG] df_formations:\n", df_formations)

# ===========================================
# == Fonctions utilitaires (matching)      ==
# ===========================================

def extract_keywords(objective: str, knowledge: str) -> List[str]:
    """
    Nettoie et extrait les mots-clés significatifs à partir 
    de l’objectif et des compétences, en supprimant les stop words.
    """
    # Liste de mots fréquents à ignorer
    stop_words = {
        "le", "la", "les", "de", "des", "du", "un", "une", "et", "à", "en", 
        "au", "aux", "pour", "avec", "dans", "sur", "par", "se", "son", 
        "sa", "ses", "ce", "cette", "ces", "est", "qui", "que", "dont", 
        "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles"
    }

    # Mise en minuscules + remplacement des virgules par des espaces
    obj_str = objective.lower().replace(",", " ")
    knw_str = knowledge.lower().replace(",", " ")

    # Tokenisation basique
    obj_tokens = obj_str.split()
    knw_tokens = knw_str.split()

    # Fusion et nettoyage
    raw_tokens = obj_tokens + knw_tokens
    cleaned_tokens = [
        t.strip() for t in raw_tokens if t.strip() and t not in stop_words
    ]

    # Suppression des doublons
    unique_tokens = list(set(cleaned_tokens))

    # print("[DEBUG] Mots-clés nettoyés (sans stop words) :", unique_tokens)

    return unique_tokens



def partial_match_formations(df: pd.DataFrame, tokens: List[str], niveau_user: str, seuil_score: int) -> pd.DataFrame:
    """
    Filtre et trie les formations en fonction de :
      - correspondances avec les mots-clés
      - niveau de l'utilisateur (bonus/malus)
      - score total >= seuil minimal

    :param df: DataFrame des formations
    :param tokens: mots-clés extraits du profil
    :param niveau_user: 'débutant', 'intermédiaire' ou 'avancé'
    :param seuil_score: score minimal requis pour être considéré
    :return: formations triées par pertinence

    Exemple :
        Les formations trop faibles (score < {seuil}) sont ignorées
        Les débutants sont orientés vers des formations plus accessibles
        Les profils avancés obtiennent des contenus plus techniques 
    """
    # print("\n[DEBUG] partial_match_formations()")
    # print(f"[DEBUG] Niveau utilisateur : {niveau_user}")
    # print(f"[DEBUG] Tokens utilisés : {tokens}")

    if df.empty or not tokens:
        # print("[WARNING] DF vide ou aucun token")
        return df.iloc[0:0]

    df = df.copy()

    # Crée un corpus global pour chaque formation
    df["corpus"] = df.apply(
        lambda row: " ".join(
            str(x).lower() for lst in [row.get("objectifs", []), row.get("prerequis", []), row.get("programme", [])]
            for x in (lst if isinstance(lst, list) else [lst])
        ),
        axis=1
    )

    # Ajoute une colonne "score" initiale
    def compute_score(row):
        text = row["corpus"]
        # Compte le nombre de tokens présents dans le texte
        score = sum(1 for t in tokens if t in text)

        # BONUS SI LE NIVEAU UTILISATEUR CORRESPOND À CELUI DE LA FORMATION
        niveau_formation = row.get("niveau", "").lower()
        prerequis = row.get("prerequis", [])

        # print(f"[DEBUG] Analyse niveau => user: {niveau_user} | formation: {niveau_formation} | prerequis: {prerequis}")

        if niveau_user == "débutant":
            # Bonus si la formation est marquée 'débutant' ou n’a aucun prérequis
            if "débutant" in niveau_formation or not prerequis:
                score += 2
        elif niveau_user == "avancé":
            # Bonus uniquement si formation = 'avancé'
            if "avancé" in niveau_formation:
                score += 1

        return score

    df["score"] = df.apply(compute_score, axis=1)

    print("\n[DEBUG] Scores détaillés :")
    print(df[["titre", "score"]].sort_values(by="score", ascending=False).to_string(index=False))

    # Ne garde que les formations qui dépassent un seuil minimum
    filtered = df[df["score"] >= seuil_score].sort_values(by="score", ascending=False)

    # print(f"[DEBUG] Formations retenues après filtrage (seuil={seuil_score}) : {len(filtered)}")
    return filtered




# ===========================================
# == Schémas Pydantic pour l'API           ==
# ===========================================

class UserProfile(BaseModel):
    name: str
    objective: str
    level: str
    knowledge: str
    recommended_course: Optional[str] = None

class RecommendRequest(BaseModel):
    profile: UserProfile

class RecommendResponse(BaseModel):
    recommended_course: str
    reply: str

class QueryRequest(BaseModel):
    profile: UserProfile
    history: List[dict] = []
    question: str

class QueryResponse(BaseModel):
    reply: str

# eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2N2VmZGRhZWMzMjc2ZTA2Y2YxZTJiOTMiLCJhcGlfa2V5IjoiV0o2aG1PdXRzOU5BTTNMbUR5eVN4RUJtdjg1MmNwNGYvaWRvcTZ5b29GST0iLCJvcmdhbmlzYXRpb24iOm51bGwsImVtYWlsIjoibWljaGVsLmdlcm1hbm90dGlAZ21haWwuY29tIiwiaWF0IjoxNzQzNzczMTEwLCJleHAiOjE3NzUzMDkxMTAsImlzcyI6ImFwaSJ9.R-_-cNlcRrczSsyBJ3nzaSoOVLkvMGhe87bVMd3ytII
# ===========================================
# == Endpoint /query                       ==
# ===========================================
@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    """
    Endpoint simulant une conversation. 
    Répond juste avec une phrase fictive.
    """
    question = req.question.strip()
    print("\n[DEBUG] /query => question =", question)
    return QueryResponse(reply=f"Réponse fictive à '{question}'. (Pas de LLM)")

# ===========================================
# == Endpoint /recommend                   ==
# ===========================================
@app.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(r: RecommendRequest):
    """
    Reçoit un profil utilisateur, extrait les mots-clés,
    puis cherche une formation correspondante.
    Retourne une réponse structurée avec détails séparés.
    """
    profile = r.profile  # alias pour plus de clarté

    # Debug
    # print(f"[DEBUG] /recommend => Profil = {profile}")

    # Extraction des mots-clés à partir de l’objectif et des connaissances
    tokens = extract_keywords(profile.objective, profile.knowledge)

    # Récupération du niveau utilisateur en minuscule
    niveau = profile.level.lower().strip()

    # Défiiner le seuil de score minimal
    seuil = 6

    # Matching dans les formations avec score et niveau pris en compte
    matched_df = partial_match_formations(df_formations, tokens, niveau_user=niveau, seuil_score=seuil)



    if not matched_df.empty:
        match = matched_df.iloc[0]

        titre = match["titre"]
        objectifs = match["objectifs"]
        prerequis = match["prerequis"]
        programme = match["programme"]
        lien = match["lien"]

        return RecommendResponse(
            recommended_course=titre,
            reply=f"Voici une formation qui correspond à votre profil.",
            details={
                "objectifs": objectifs,
                "prerequis": prerequis,
                "programme": programme,
                "lien": lien
            }
        )
    # Si aucune formation ne correspond, on renvoie un message générique
    else:
        return RecommendResponse(
            recommended_course="Aucune formation pertinente",
            reply="Aucune formation ne correspond aux mots-clés fournis.",
            details=None
        )

