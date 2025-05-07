# app/utils.py
"""
Utilitaires centralisés pour l'application de chatbot de formation.
Contient des fonctions d'assainissement d'entrées, de gestion de fichiers
et un service de données partagé.
"""

import html
import json
import logging
import re
import time
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import app.globals as globs
engine = globs.llm_engine

logger = logging.getLogger(__name__)


def search_and_format_courses(query: str, k: int = 5):
    fs = globs.formation_search

    results = fs.search(query, k=k)

    internal = [r for r in results if r[0].get("_source") == "internal"]
    external = [r for r in results if r[0].get("_source") != "internal"]

    output = ""
    if internal:
        output += f"💼 {len(internal)} formation(s) interne(s) trouvée(s) :\n"
        output += "Voici des formations proposées par Beyond Expertise :\n"
        for meta, score in internal:
            output += f"\n {meta['titre']}\n"
            #output += f"  ID: {meta.get('ID', 'N/A')} — Score: {score:.3f}\n"
            output += f"Objectifs : {meta.get('CAPACITES_ATTESTEES', 'non renseigné')}  \n"
            output += f"Programme : {meta.get('ACTIVITES_VISEES', 'non renseigné')}  \n"
            output += f"Emploi accessibles après la formation : {meta.get('TYPE_EMPLOI_ACCESSIBLES', 'non renseigné')}  \n"
            output += f"Public visé : {meta.get('public', '')}  \n"
            output += f"Prérequis : {meta.get('prerequis', '')}  \n"
            output += f"\n\nAutre infos secondaires si l'utilisateur le demande :\n"
            output += f"  Lieu : {meta.get('lieu', '')}\n"
            output += f"  Tarif : {meta.get('tarif', '')}\n"
            output += f"  Durée : {meta.get('duree', '')}\n"
            output += f"  Certifiante : {meta.get('certifiant', '')}\n"
            output += f"  Modalité : {meta.get('modalite', '')}\n"
    elif external:
        # Filtrer les formations avec score > 0.30
        valid_results = [(meta, score) for meta, score in external if score > 0.30]

        if valid_results:
            output += f"📚 {len(valid_results)} formation(s) RNCP trouvée(s) :\n"
            output += "Voici des formations du RNCP que j'ai trouvé pour toi :\n"
            for meta, score in valid_results:
                output += f"Score : {score}"
                output += f"\n {meta['titre']}\n"
                output += f" ID  : {meta.get('ID', 'N/A')}\n"
                output += f" Programme : {meta.get('ACTIVITES_VISEES', 'N/A')}\n"
                output += f" Objectifs : {meta.get('CAPACITES_ATTESTEES', 'N/A')}\n"
                output += f" URL(s) : {extract_urls_from_text(meta.get('LIEN_URL_DESCRIPTION', 'N/A'))}\n"
                output += f"  Certificateur  : {meta.get('CERTIFICATEURS', 'N/A')}\n"
                output += f"  Niveau : {meta.get('ABREGE_LIBELLES', '')}\n"
                output += f"  Emplois : {meta.get('TYPE_EMPLOI_ACCESSIBLES', '')}\n"
        else:
            #output += "Aucune formation RNCP pertinente trouvée."
            False

    else:   
        #output = "Aucune formation trouvée."
        False

    source_type = "internal" if internal else "external"
    print(f"\n\nRésultats : \n{output}\n\n")
    return output.strip(), source_type

# --------------------------------------------------
# Fonctions d'assainissement des entrées
# --------------------------------------------------
def extract_urls_from_text(text):
    url_pattern = re.compile(r'https?:\/\/[^\s\[\]\n]+')
    urls = url_pattern.findall(text)
    for url in urls:
        print(url)

def normalize_course(course: dict) -> dict:
    """
    Harmonise les clés et complète les valeurs manquantes d'un cours.
    - corrige les variantes d'orthographe ('duree' ➜ 'durée', etc.)
    - force la présence d'un champ 'id' et d'un champ 'certifiant'
    """
    if not isinstance(course, dict):
        return {}

    mapping = {
        "duree": "durée",
        "duration": "durée",
        "modalite": "modalité",
        "title": "titre"
    }
    for old, new in mapping.items():
        if old in course and new not in course:
            course[new] = course.pop(old)
    
    # Champs obligatoires avec valeurs par défaut
    course.setdefault("id", course.get("titre", "").lower().replace(" ", "_"))
    course.setdefault("certifiant", False)
    course.setdefault("durée", "Non spécifiée")
    course.setdefault("modalité", "Non spécifiée")
    course.setdefault("lieu", "Non spécifié")
    course.setdefault("prochaines_sessions", "Non spécifiées")

    return course


