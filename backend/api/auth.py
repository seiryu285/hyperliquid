"""
Authentication API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from models.user import UserCreate, User, Token
from services.auth_service import AuthService
from core.dependencies import get_auth_service, get_current_user

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/register", response_model=User)
async def register(
    user_create: UserCreate,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Register a new user."""
    try:
        return await auth_service.create_user(user_create)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Authenticate user and return tokens."""
    user = await auth_service.authenticate_user(
        form_data.username,  # OAuth2 form uses username field for email
        form_data.password
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled"
        )
    
    # Create tokens
    access_token = auth_service.create_access_token(user)
    refresh_token = auth_service.create_refresh_token(user)
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token
    )

@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_token: str,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Get new access token using refresh token."""
    try:
        return await auth_service.refresh_token(refresh_token)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Logout current user."""
    # Optional: Implement token blacklisting here
    return {"message": "Successfully logged out"}

@router.get("/me", response_model=User)
async def get_current_user(
    current_user: User = Depends(get_current_user)
):
    """Get current authenticated user.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Current user details
    """
    return current_user
