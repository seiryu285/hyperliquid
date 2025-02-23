"""
Security-related API endpoints for password reset and 2FA.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict

from services.security_service import SecurityService
from services.auth_service import AuthService
from models.user import (
    PasswordReset,
    PasswordResetConfirm,
    TwoFactorSetup,
    TwoFactorVerify,
    User
)
from core.dependencies import get_security_service, get_current_user

router = APIRouter(prefix="/security", tags=["security"])

@router.post("/password-reset/request")
async def request_password_reset(
    reset_request: PasswordReset,
    security_service: SecurityService = Depends(get_security_service)
):
    """Request password reset email.
    
    Args:
        reset_request: Password reset request
        security_service: Security service instance
    """
    await security_service.initiate_password_reset(reset_request.email)
    return {
        "message": "If an account exists with this email, "
                  "a password reset link has been sent."
    }

@router.post("/password-reset/confirm")
async def confirm_password_reset(
    reset_confirm: PasswordResetConfirm,
    security_service: SecurityService = Depends(get_security_service)
):
    """Confirm password reset with token.
    
    Args:
        reset_confirm: Password reset confirmation
        security_service: Security service instance
    """
    success = await security_service.complete_password_reset(
        reset_confirm.token,
        reset_confirm.new_password
    )
    
    if success:
        return {"message": "Password has been reset successfully"}
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Password reset failed"
    )

@router.post("/2fa/setup", response_model=TwoFactorSetup)
async def setup_2fa(
    current_user: User = Depends(get_current_user),
    security_service: SecurityService = Depends(get_security_service)
):
    """Set up 2FA for current user.
    
    Args:
        current_user: Current authenticated user
        security_service: Security service instance
        
    Returns:
        2FA setup information
    """
    if current_user.is_2fa_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="2FA is already enabled"
        )
    
    return await security_service.setup_2fa(current_user.id)

@router.post("/2fa/verify")
async def verify_2fa(
    verify_request: TwoFactorVerify,
    current_user: User = Depends(get_current_user),
    security_service: SecurityService = Depends(get_security_service),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Verify 2FA token and complete authentication.
    
    Args:
        verify_request: 2FA verification request
        current_user: Current authenticated user
        security_service: Security service instance
        auth_service: Authentication service instance
        
    Returns:
        New access token with 2FA verification
    """
    if not current_user.is_2fa_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="2FA is not enabled"
        )
    
    if await security_service.verify_2fa(current_user.id, verify_request.token):
        # Generate new token with 2FA verification
        return await auth_service.create_2fa_verified_token(current_user)
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid 2FA token"
    )

@router.post("/2fa/disable")
async def disable_2fa(
    verify_request: TwoFactorVerify,
    current_user: User = Depends(get_current_user),
    security_service: SecurityService = Depends(get_security_service)
):
    """Disable 2FA for current user.
    
    Args:
        verify_request: 2FA verification request
        current_user: Current authenticated user
        security_service: Security service instance
    """
    if not current_user.is_2fa_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="2FA is not enabled"
        )
    
    success = await security_service.disable_2fa(
        current_user.id,
        verify_request.token
    )
    
    if success:
        return {"message": "2FA has been disabled successfully"}
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid 2FA token"
    )
