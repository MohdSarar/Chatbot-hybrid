# main.py
"""
Point d'entrée de l'API Chatbot Formation.
Initialisation FastAPI + Montage des routes + CORS + Chargement .env.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pathlib import Path

# Import des routes éclatées
from app.routes.recommend_routes import router as recommend_router
from app.routes.query_routes import router as query_router
from app.routes.upload_routes import router as upload_router
from app.routes.email_routes import router as email_router

from app.logging_config import logger

# Chargement .env
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
print("[INFO] Variables d'environnement chargées (.env)")
logger.info("Variables d'environnement chargées depuis .env")

# Initialisation FastAPI
app = FastAPI(
    title="Chatbot Formation API (Pandas + dossier content)",
    version="1.0.0",
    description="API de recommandation de formations utilisant pandas, avec chargement JSON depuis 'content'."
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montage des routes FastAPI
app.include_router(recommend_router)
app.include_router(query_router)
app.include_router(upload_router)
app.include_router(email_router)

print("[INFO] API FastAPI initialisée et routes montées avec succès.")
logger.info("API FastAPI initialisée et routes montées.")
