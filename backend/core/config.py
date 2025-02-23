"""
Application configuration settings.
"""

from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database settings
    MONGO_URL: str = "mongodb://localhost:27017"
    MONGO_DB: str = "hyperliquid"
    
    # JWT settings
    JWT_SECRET_KEY: str = "your-secret-key-here"  # Change in production
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Email settings
    MAIL_USERNAME: str = ""
    MAIL_PASSWORD: str = ""
    MAIL_FROM: str = ""
    MAIL_PORT: int = 587
    MAIL_SERVER: str = "smtp.gmail.com"
    
    # Frontend URL for password reset
    FRONTEND_URL: str = "http://localhost:3000"
    
    # Security settings
    PASSWORD_RESET_TOKEN_EXPIRE_HOURS: int = 24
    MAX_LOGIN_ATTEMPTS: int = 5
    ACCOUNT_LOCKOUT_MINUTES: int = 15
    
    class Config:
        env_file = ".env"

settings = Settings()
