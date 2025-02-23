"""
Unit tests for security service.
"""

import pytest
from datetime import datetime, timedelta
from fastapi import HTTPException
import pyotp
import base64
from PIL import Image
import io

from backend.models.user import UserCreate, UserInDB
from backend.services.security_service import SecurityService

pytestmark = pytest.mark.asyncio

async def test_initiate_password_reset(
    security_service: SecurityService,
    test_user: UserInDB
):
    """Test password reset initiation."""
    # Test with existing user
    result = await security_service.initiate_password_reset(test_user.email)
    assert result is True
    
    # Verify token was created
    user = await security_service.db.users.find_one({"email": test_user.email})
    assert user["password_reset_token"] is not None
    assert user["password_reset_expires"] > datetime.utcnow()
    
    # Test with non-existent user (should still return True for security)
    result = await security_service.initiate_password_reset("nonexistent@example.com")
    assert result is True

async def test_complete_password_reset(
    security_service: SecurityService,
    test_user: UserInDB
):
    """Test password reset completion."""
    # First initiate reset
    await security_service.initiate_password_reset(test_user.email)
    user = await security_service.db.users.find_one({"email": test_user.email})
    token = user["password_reset_token"]
    
    # Complete reset
    new_password = "NewTest123!@#"
    result = await security_service.complete_password_reset(token, new_password)
    assert result is True
    
    # Verify password was changed
    updated_user = await security_service.db.users.find_one({"email": test_user.email})
    assert updated_user["password_reset_token"] is None
    assert updated_user["password_reset_expires"] is None
    
    # Try to authenticate with new password
    from backend.services.auth_service import AuthService
    auth_service = AuthService(security_service.db_client)
    authenticated_user = await auth_service.authenticate_user(
        test_user.email,
        new_password
    )
    assert authenticated_user is not None

async def test_complete_password_reset_invalid_token(
    security_service: SecurityService
):
    """Test password reset with invalid token."""
    with pytest.raises(HTTPException) as exc_info:
        await security_service.complete_password_reset(
            "invalid_token",
            "NewTest123!@#"
        )
    assert exc_info.value.status_code == 400
    assert "Invalid or expired reset token" in str(exc_info.value.detail)

async def test_setup_2fa(security_service: SecurityService, test_user: UserInDB):
    """Test 2FA setup."""
    setup = await security_service.setup_2fa(test_user.id)
    
    # Verify response format
    assert "secret" in setup
    assert "qr_code" in setup
    assert "backup_codes" in setup
    assert len(setup["backup_codes"]) == 10
    
    # Verify QR code is valid base64 image
    qr_data = base64.b64decode(setup["qr_code"])
    image = Image.open(io.BytesIO(qr_data))
    assert image.format == "PNG"
    
    # Verify user was updated
    user = await security_service.db.users.find_one({"_id": test_user.id})
    assert user["totp_secret"] == setup["secret"]
    assert user["is_2fa_enabled"] is True
    assert len(user["backup_codes"]) == 10

async def test_verify_2fa(security_service: SecurityService, test_user: UserInDB):
    """Test 2FA verification."""
    # Setup 2FA first
    setup = await security_service.setup_2fa(test_user.id)
    
    # Generate valid token
    totp = pyotp.TOTP(setup["secret"])
    token = totp.now()
    
    # Verify token
    result = await security_service.verify_2fa(test_user.id, token)
    assert result is True
    
    # Test invalid token
    result = await security_service.verify_2fa(test_user.id, "000000")
    assert result is False

async def test_verify_2fa_backup_code(
    security_service: SecurityService,
    test_user: UserInDB
):
    """Test 2FA verification with backup code."""
    # Setup 2FA first
    setup = await security_service.setup_2fa(test_user.id)
    backup_code = setup["backup_codes"][0]
    
    # Verify backup code
    result = await security_service.verify_2fa(test_user.id, backup_code)
    assert result is True
    
    # Verify backup code was removed
    user = await security_service.db.users.find_one({"_id": test_user.id})
    assert backup_code not in [
        code for code in user["backup_codes"]
    ]

async def test_disable_2fa(security_service: SecurityService, test_user: UserInDB):
    """Test 2FA disabling."""
    # Setup 2FA first
    setup = await security_service.setup_2fa(test_user.id)
    
    # Generate valid token
    totp = pyotp.TOTP(setup["secret"])
    token = totp.now()
    
    # Disable 2FA
    result = await security_service.disable_2fa(test_user.id, token)
    assert result is True
    
    # Verify user was updated
    user = await security_service.db.users.find_one({"_id": test_user.id})
    assert user["is_2fa_enabled"] is False
    assert user["totp_secret"] is None
    assert len(user["backup_codes"]) == 0

async def test_disable_2fa_invalid_token(
    security_service: SecurityService,
    test_user: UserInDB
):
    """Test 2FA disabling with invalid token."""
    # Setup 2FA first
    await security_service.setup_2fa(test_user.id)
    
    with pytest.raises(HTTPException) as exc_info:
        await security_service.disable_2fa(test_user.id, "000000")
    assert exc_info.value.status_code == 400
    assert "Invalid 2FA token" in str(exc_info.value.detail)
