"""
FastAPI dependencies for authentication and services.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional

from core.config import settings
from services.auth_service import AuthService
from services.security_service import SecurityService
from models.user import User

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Database client
client = AsyncIOMotorClient(settings.MONGO_URL)

def get_auth_service() -> AuthService:
    """Get AuthService instance."""
    return AuthService(client)

def get_security_service() -> SecurityService:
    """Get SecurityService instance."""
    return SecurityService(client)

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    auth_service: AuthService = Depends(get_auth_service)
) -> User:
    """Get current authenticated user."""
    return await auth_service.get_current_user(token)

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user

async def get_2fa_verified_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current user with 2FA verification."""
    if current_user.is_2fa_enabled and not getattr(current_user, "is_2fa_verified", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="2FA verification required"
        )
    return current_user
