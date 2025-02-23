"""
Unit tests for authentication service.
"""

import pytest
from datetime import datetime, timedelta
from fastapi import HTTPException
import jwt

from backend.models.user import UserCreate, UserInDB
from backend.services.auth_service import AuthService
from backend.core.config import settings

pytestmark = pytest.mark.asyncio

async def test_create_user(auth_service: AuthService):
    """Test user creation."""
    user_create = UserCreate(
        email="test@example.com",
        username="testuser",
        password="Test123!@#"
    )
    
    user = await auth_service.create_user(user_create)
    assert user.email == user_create.email
    assert user.username == user_create.username
    assert not hasattr(user, "password")
    assert not hasattr(user, "hashed_password")

async def test_create_duplicate_user(auth_service: AuthService, test_user: UserInDB):
    """Test duplicate user creation."""
    user_create = UserCreate(
        email=test_user.email,
        username="newuser",
        password="Test123!@#"
    )
    
    with pytest.raises(HTTPException) as exc_info:
        await auth_service.create_user(user_create)
    assert exc_info.value.status_code == 400
    assert "Email already registered" in str(exc_info.value.detail)

async def test_authenticate_user(auth_service: AuthService, test_user: UserInDB):
    """Test user authentication."""
    user = await auth_service.authenticate_user(
        test_user.email,
        "Test123!@#"
    )
    assert user is not None
    assert user.email == test_user.email

async def test_authenticate_user_invalid_password(
    auth_service: AuthService,
    test_user: UserInDB
):
    """Test authentication with invalid password."""
    user = await auth_service.authenticate_user(
        test_user.email,
        "WrongPassword123!"
    )
    assert user is None

async def test_create_access_token(auth_service: AuthService, test_user: UserInDB):
    """Test access token creation."""
    token = auth_service.create_access_token(test_user)
    payload = jwt.decode(
        token,
        settings.JWT_SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM]
    )
    
    assert payload["sub"] == str(test_user.id)
    assert payload["type"] == "access"
    assert "exp" in payload

async def test_create_refresh_token(auth_service: AuthService, test_user: UserInDB):
    """Test refresh token creation."""
    token = auth_service.create_refresh_token(test_user)
    payload = jwt.decode(
        token,
        settings.JWT_SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM]
    )
    
    assert payload["sub"] == str(test_user.id)
    assert payload["type"] == "refresh"
    assert "exp" in payload

async def test_get_current_user(
    auth_service: AuthService,
    test_user: UserInDB,
    test_user_token: str
):
    """Test getting current user from token."""
    user = await auth_service.get_current_user(test_user_token)
    assert user.id == test_user.id
    assert user.email == test_user.email

async def test_get_current_user_invalid_token(auth_service: AuthService):
    """Test getting current user with invalid token."""
    with pytest.raises(HTTPException) as exc_info:
        await auth_service.get_current_user("invalid_token")
    assert exc_info.value.status_code == 401

async def test_get_current_user_expired_token(
    auth_service: AuthService,
    test_user: UserInDB
):
    """Test getting current user with expired token."""
    # Create token that expired 1 hour ago
    expired_token = auth_service.create_access_token(
        test_user,
        expires_delta=timedelta(hours=-1)
    )
    
    with pytest.raises(HTTPException) as exc_info:
        await auth_service.get_current_user(expired_token)
    assert exc_info.value.status_code == 401
    assert "Token has expired" in str(exc_info.value.detail)

async def test_refresh_token(
    auth_service: AuthService,
    test_user: UserInDB
):
    """Test token refresh."""
    refresh_token = auth_service.create_refresh_token(test_user)
    result = await auth_service.refresh_token(refresh_token)
    
    assert "access_token" in result
    assert result["token_type"] == "bearer"
    
    # Verify new access token
    payload = jwt.decode(
        result["access_token"],
        settings.JWT_SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM]
    )
    assert payload["sub"] == str(test_user.id)
    assert payload["type"] == "access"

async def test_refresh_token_invalid(auth_service: AuthService):
    """Test token refresh with invalid token."""
    with pytest.raises(HTTPException) as exc_info:
        await auth_service.refresh_token("invalid_token")
    assert exc_info.value.status_code == 401
    assert "Invalid refresh token" in str(exc_info.value.detail)
