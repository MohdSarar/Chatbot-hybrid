# app/logging_config.py
"""
Configuration du logger global pour l'application.
Permet d'afficher les logs dans la console avec formatage.
"""

import logging

# Création d'un logger nommé "app"
logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG)  # Niveau global : DEBUG pour tout voir

# Création d'un handler pour afficher les logs dans la console
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

handler.setFormatter(formatter)
logger.addHandler(handler)
