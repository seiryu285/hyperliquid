"""
Authentication service for handling user authentication and token management.
"""

from datetime import datetime, timedelta
from typing import Optional
import jwt
from fastapi import HTTPException, status
from motor.motor_asyncio import AsyncIOMotorClient
import bcrypt
from passlib.context import CryptContext

from models.user import UserInDB, User
from core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    def __init__(self, db_client: AsyncIOMotorClient):
        self.db = db_client[settings.MONGO_DB]
        self.users_collection = self.db.users

    async def authenticate_user(self, email: str, password: str) -> Optional[UserInDB]:
        user = await self.users_collection.find_one({"email": email})
        if not user:
            return None

        user_db = UserInDB(**user)
        if not pwd_context.verify(password, user_db.hashed_password):
            return None

        return user_db

    def create_access_token(self, user: User, is_2fa_verified: bool = False) -> str:
        expires_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        expire = datetime.utcnow() + expires_delta
        
        to_encode = {
            "sub": str(user.id),
            "exp": expire,
            "type": "access",
            "is_2fa_verified": is_2fa_verified
        }
        
        return jwt.encode(
            to_encode,
            settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM
        )

    def create_refresh_token(self, user: User) -> str:
        expires_delta = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        expire = datetime.utcnow() + expires_delta
        
        to_encode = {
            "sub": str(user.id),
            "exp": expire,
            "type": "refresh"
        }
        
        return jwt.encode(
            to_encode,
            settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM
        )

    async def get_current_user(self, token: str) -> User:
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM]
            )
            user_id = payload.get("sub")
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials"
                )
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )

        user = await self.users_collection.find_one({"_id": user_id})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        return User(**user)

    async def create_2fa_verified_token(self, user: User) -> dict:
        access_token = self.create_access_token(user, is_2fa_verified=True)
        refresh_token = self.create_refresh_token(user)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }

    async def refresh_token(self, refresh_token: str) -> dict:
        try:
            payload = jwt.decode(
                refresh_token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM]
            )
            if payload.get("type") != "refresh":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            user = await self.get_current_user(refresh_token)
            access_token = self.create_access_token(user)
            
            return {
                "access_token": access_token,
                "token_type": "bearer"
            }
        except (jwt.ExpiredSignatureError, jwt.JWTError):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
