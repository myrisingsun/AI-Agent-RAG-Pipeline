from typing import Any

from fastapi import HTTPException, Request, status
from jose import JWTError, jwt

from src.common.logging import get_logger

logger = get_logger(__name__)

_DEV_USER: dict[str, Any] = {"sub": "dev-user", "role": "analyst"}


async def require_auth(request: Request) -> dict[str, Any]:
    """
    FastAPI dependency: validate JWT from Authorization header.
    If jwt_secret == "dev", skip validation and return a mock user.
    """
    config = request.app.state.config
    if config.jwt_secret == "dev":
        return _DEV_USER

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_header.removeprefix("Bearer ")
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            config.jwt_secret,
            algorithms=[config.jwt_algorithm],
        )
        return payload
    except JWTError as exc:
        logger.warning("JWT validation failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
