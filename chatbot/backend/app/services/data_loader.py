# app/services/data_loader.py
"""
Service de chargement des fichiers JSON de formations dans un DataFrame pandas.
"""

import pandas as pd
import json
from pathlib import Path
from app.logging_config import logger

def load_formations_to_df(json_dir: Path) -> pd.DataFrame:
    """
    Parcourt tous les fichiers *.json dans json_dir,
    renvoie un DataFrame (titre, objectifs, prérequis, etc.)
    """
    if not json_dir.exists():
        logger.warning("Le dossier %s n'existe pas.", json_dir)
        print(f"[WARNING] Le dossier {json_dir} n'existe pas.")
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
                    "lien": data.get("lien", ""),
                    "durée": data.get("durée", ""),
                    "tarif": data.get("tarif", ""),
                    "modalité": data.get("modalité", ""),
                    "certifiant": data.get("certifiant"),
                })
                logger.debug("Fichier chargé : %s", file.name)
        except Exception as e:
            logger.error("Erreur de lecture du fichier %s : %s", file.name, str(e))
            print(f"[ERROR] Erreur lecture fichier {file.name} : {e}")

    formations_df = pd.DataFrame(records)
    logger.info("%d formations chargées depuis %s", len(formations_df), json_dir)
    print(f"[INFO] {len(formations_df)} formations chargées depuis {json_dir}")
    return formations_df
