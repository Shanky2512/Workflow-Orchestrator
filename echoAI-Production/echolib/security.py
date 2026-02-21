"""
EchoAI Security Module

Provides JWT-based authentication and authorization for the EchoAI platform.

Auth Enforcement Modes (via AUTH_ENFORCEMENT setting):
    - "optional": Unauthenticated requests get a default anonymous context
    - "required": All requests must have valid JWT (401 on missing/invalid token)

Usage:
    # Required auth (raises 401 if no valid token)
    @router.get('/protected')
    async def protected(user: UserContext = Depends(require_user)):
        ...

    # Optional auth (returns anonymous context if no token)
    @router.get('/public')
    async def public(user: UserContext = Depends(get_current_user)):
        ...
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
import logging
import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .config import settings
from .types import UserContext

logger = logging.getLogger(__name__)

# HTTP Bearer security scheme (auto_error=False allows optional auth)
security = HTTPBearer(auto_error=False)

# Anonymous user context for optional auth mode
ANONYMOUS_USER_ID = "anonymous"
ANONYMOUS_EMAIL = "anonymous@echo.local"


class AuthError(HTTPException):
    """Authentication error exception."""

    def __init__(self, detail: str = "Unauthorized"):
        super().__init__(status_code=401, detail=detail)


def create_token(sub: str, email: str, *, expires_minutes: int = 60) -> str:
    """
    Create a JWT token.

    Args:
        sub: User ID (subject claim)
        email: User email
        expires_minutes: Token expiration time in minutes

    Returns:
        Encoded JWT token string
    """
    now = datetime.now(tz=timezone.utc)
    payload = {
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expires_minutes)).timestamp()),
        "sub": sub,
        "email": email,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_token(token: str) -> Optional[dict]:
    """
    Decode and validate a JWT token.

    Args:
        token: JWT token string

    Returns:
        Decoded payload dict if valid, None otherwise
    """
    try:
        return jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
        )
    except jwt.ExpiredSignatureError:
        logger.debug("Token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.debug(f"Invalid token: {e}")
        return None
    except Exception as e:
        logger.warning(f"Token decode error: {e}")
        return None


def _extract_user_from_token(
    creds: Optional[HTTPAuthorizationCredentials],
) -> Optional[UserContext]:
    """
    Extract user context from HTTP Authorization credentials.

    Args:
        creds: HTTP Bearer credentials (may be None)

    Returns:
        UserContext if valid token, None otherwise
    """
    if creds is None:
        return None

    payload = decode_token(creds.credentials)
    if not payload:
        return None

    return UserContext(
        user_id=payload.get("sub", ""),
        email=payload.get("email", ""),
    )


async def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> UserContext:
    """
    Get current user with optional authentication.

    Behavior depends on AUTH_ENFORCEMENT setting:
        - "optional": Returns anonymous context if no valid token
        - "required": Raises 401 if no valid token

    Args:
        creds: HTTP Bearer credentials (injected by FastAPI)

    Returns:
        UserContext for authenticated or anonymous user

    Raises:
        AuthError: If auth is required and no valid token provided
    """
    user = _extract_user_from_token(creds)

    if user is not None:
        return user

    # No valid token - check enforcement mode
    if settings.auth_enforcement == "required":
        raise AuthError("Authentication required")

    # Optional mode - return anonymous user
    return UserContext(user_id=ANONYMOUS_USER_ID, email=ANONYMOUS_EMAIL)


async def require_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> UserContext:
    """
    Require authenticated user (always enforces auth regardless of setting).

    Use this for endpoints that must always require authentication,
    regardless of the global AUTH_ENFORCEMENT setting.

    Args:
        creds: HTTP Bearer credentials (injected by FastAPI)

    Returns:
        UserContext for authenticated user

    Raises:
        AuthError: If no valid token provided
    """
    user = _extract_user_from_token(creds)

    if user is None:
        raise AuthError("Authentication required")

    return user


# Legacy alias for backward compatibility
async def user_context(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> UserContext:
    """
    Legacy user context dependency.

    Deprecated: Use get_current_user or require_user instead.
    This function now delegates to get_current_user for backward compatibility.
    """
    return await get_current_user(creds)


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for FastAPI.

    Extracts user from JWT token and adds to request state.
    Does NOT block unauthenticated requests - that's handled by route dependencies.

    Usage:
        app.add_middleware(AuthMiddleware)

    Then in routes:
        request.state.user  # UserContext or None
    """

    # Paths that skip auth processing entirely
    SKIP_PATHS = frozenset(["/healthz", "/docs", "/openapi.json", "/redoc"])

    async def dispatch(self, request: Request, call_next):
        """Process request and extract user context."""
        path = request.url.path

        # Skip auth for health checks and docs
        if path in self.SKIP_PATHS:
            return await call_next(request)

        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization")
        user = None

        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            payload = decode_token(token)
            if payload:
                user = UserContext(
                    user_id=payload.get("sub", ""),
                    email=payload.get("email", ""),
                )

        # Attach user to request state (may be None)
        request.state.user = user

        # Continue to route handler
        return await call_next(request)


def get_user_from_request(request: Request) -> Optional[UserContext]:
    """
    Get user from request state (set by AuthMiddleware).

    Args:
        request: FastAPI Request object

    Returns:
        UserContext if user is authenticated, None otherwise
    """
    return getattr(request.state, "user", None)
