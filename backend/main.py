"""
main.py
───────
WATRS v2.0 — Weather-Aware Tour Recommendation Engine
FastAPI application entry point.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient

from api.v1.recommendations import router as recommendations_router
from core.config import get_settings
from core.database import initialize_db
from core.security import setup_security

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("watrs")

# ── Settings ────────────────────────────────────────────────────────────────
settings = get_settings()


# ── Lifespan (startup / shutdown) ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the MongoDB connection pool across the app lifetime."""
    logger.info("Connecting to MongoDB …")
    app.state.mongo_client = AsyncIOMotorClient(settings.MONGODB_URL)
    app.state.db = app.state.mongo_client.get_default_database("watrs_db")
    logger.info("MongoDB connected ✓")

    # Ensure required indexes exist (idempotent)
    await initialize_db(app.state.db)
    logger.info("Database initialization complete ✓")

    yield  # ← application runs here

    logger.info("Shutting down MongoDB connection …")
    app.state.mongo_client.close()
    logger.info("MongoDB disconnected ✓")


# ── App factory ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="WATRS v2.0",
    version="2.0.0",
    description="Weather-Aware Tour Recommendation Engine — API",
    lifespan=lifespan,
)

# Wire security middleware (rate limiter, headers, CORS, exception handler)
setup_security(app)

# ── Routers ─────────────────────────────────────────────────────────────────
app.include_router(recommendations_router, prefix="/api/v1")


# ── Health check ────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
async def health_check():
    """Lightweight liveness probe."""
    return {"status": "healthy"}
