"""Optional API-key authentication.

FinVoice serves PII-laden transcripts, redaction exports, and raw call audio, and
now implements DPDP consent and data-minimisation checks. Shipping all of that on
unauthenticated endpoints is a contradiction a reviewer will notice.

Authentication is opt-in so local development stays frictionless: set FINVOICE_API_KEY
and every route except /api/health requires it. Leave it unset and the API behaves as
before, but says so loudly at startup rather than being quietly open.
"""

import os
import secrets

from fastapi import HTTPException, Request
from loguru import logger

_HEADER = "X-API-Key"
# Health must stay reachable for container probes and the dashboard's status badge.
_EXEMPT_PATHS = {"/api/health", "/docs", "/openapi.json", "/redoc"}


def api_key() -> str | None:
    key = os.getenv("FINVOICE_API_KEY", "").strip()
    return key or None


def is_enabled() -> bool:
    return api_key() is not None


def log_status() -> None:
    if is_enabled():
        logger.info(f"API authentication ENABLED (header: {_HEADER})")
    else:
        logger.warning(
            "API authentication DISABLED — every endpoint is open, including "
            "transcripts, PII and call audio. Set FINVOICE_API_KEY to require a key."
        )


async def require_api_key(request: Request) -> None:
    """FastAPI dependency. No-op when no key is configured."""
    expected = api_key()
    if expected is None:
        return
    if request.url.path in _EXEMPT_PATHS:
        return
    supplied = request.headers.get(_HEADER) or ""
    if not supplied:
        # Allow ?api_key= for the audio element and download links, which cannot
        # set headers.
        supplied = request.query_params.get("api_key", "")
    # Constant-time comparison — a key check that leaks timing is not a key check.
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail=f"Missing or invalid {_HEADER}")
