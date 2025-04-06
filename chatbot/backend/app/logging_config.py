# logging_config.py

import logging
from pathlib import Path

# Détermination du dossier de logs
log_dir = Path(__file__).resolve().parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

# Fichier de log
log_file = log_dir / "chatbot.log"

# Format du log
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# Handler fichier
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

# Handler console
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

# Logger global
logger = logging.getLogger("chatbot_logger")
logger.setLevel(logging.DEBUG)

# Ajouter les handlers (évite les doublons si redémarrage live)
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
