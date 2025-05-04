import os
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Routes
from app.routes.recommend_routes import router as recommend_router
from app.routes.query_routes import router as query_router
from app.routes.upload_routes import router as upload_router
from app.routes.email_routes import router as email_router

# Outils
from app.utils import DataService
from app.llm_engine import LLMEngine
from app.logging_config import logger
import app.globals as globs        # ← on importe le MODULE

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "content"

load_dotenv(APP_DIR / ".env")
logger.info(".env chargé")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Création et nettoyage des instances partagées."""
    enable_rncp = globs.enable_rncp
    enable_rncp = True  # for testing purposes

    # 1 — une seule instance DataService
    globs.data_service = DataService(
        content_dir=str(DATA_DIR),
        disable_auto_update=True,
        enable_rncp=enable_rncp,
    )

    # 2 — une seule instance LLMEngine qui RÉUTILISE ce service
    globs.llm_engine = LLMEngine(
        content_dir=str(DATA_DIR),
        disable_auto_update=True,
        enable_rncp=enable_rncp,
        data_service=globs.data_service,
    )
    globs.data_service.check_updates(force_check=True)
    globs.llm_engine._get_chroma_store(force_init=True)

    yield

    # Nettoyage
    globs.llm_engine = None
    globs.data_service = None
    logger.info("Application arrêtée")

app = FastAPI(
    title="Chatbot Formation API",
    version="1.2.0",
    description="Recommandation de formations Beyond Expertise",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montage des routes
app.include_router(recommend_router)
app.include_router(query_router)
app.include_router(upload_router)
app.include_router(email_router)
print("[INFO] API FastAPI initialisée et routes montées avec succès.")
logger.info("API FastAPI initialisée et routes montées.")