"""
core/security.py
────────────────
Security middleware and exception handlers for WATRS v2.0.

Covers:
  • Rate limiting  — slowapi with Redis backend
  • Secure headers — HSTS, X-Frame-Options, CSP via the `secure` library
  • CORS           — FastAPI CORSMiddleware
  • Exception gate — generic 500 responses to prevent info leakage
"""

import logging
from typing import TYPE_CHECKING

import secure
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from core.config import get_settings

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger("watrs.security")

# ── Settings ────────────────────────────────────────────────────────────────
settings = get_settings()

# ── Rate Limiter ────────────────────────────────────────────────────────────
# Global default: 100 requests / minute per client IP.
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/minute"],
    storage_uri=settings.REDIS_URL,
)

# Stricter limit for endpoints that call external APIs (Google / OpenWeather).
EXTERNAL_API_LIMIT = "10/minute"

# ── Secure Headers ──────────────────────────────────────────────────────────
_csp = secure.ContentSecurityPolicy().default_src("'self'")
_hsts = secure.StrictTransportSecurity().max_age(31536000).include_subdomains()
_xfo = secure.XFrameOptions().deny()

secure_headers = secure.Secure(
    csp=_csp,
    hsts=_hsts,
    xfo=_xfo,
)


# ═══════════════════════════════════════════════════════════════════════════
# Public setup function — called once from main.py
# ═══════════════════════════════════════════════════════════════════════════


def setup_security(app: "FastAPI") -> None:
    """Wire every security layer into the FastAPI application."""

    # ── 1. Rate limiting ────────────────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # ── 2. CORS ─────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # ── 3. Secure headers (HSTS / X-Frame-Options / CSP) ────────────────
    @app.middleware("http")
    async def _set_secure_headers(request: Request, call_next):  # noqa: ANN001
        response = await call_next(request)
        secure_headers.framework.fastapi(response)
        return response

    # ── 4. Global exception handler — suppress internals ────────────────
    @app.exception_handler(Exception)
    async def _global_exception_handler(request: Request, exc: Exception):  # noqa: ANN001, ARG001
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred. Please try again later."},
        )
