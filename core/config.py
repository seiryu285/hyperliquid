"""
Application configuration settings.
"""

import os
from pathlib import Path
from typing import Optional
from functools import lru_cache

from pydantic import BaseModel, Field

class Settings(BaseModel):
    """Application settings."""
    
    # Project paths
    PROJECT_ROOT: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    MODEL_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "models")
    
    # Database settings
    MONGODB_URL: str = Field(default="mongodb://localhost:27017")
    MONGO_DB: str = Field(default="hyperliquid")
    
    # Redis settings
    REDIS_URL: str = Field(default="redis://localhost:6379")
    REDIS_MAX_CONNECTIONS: int = Field(default=10)
    REDIS_SOCKET_TIMEOUT: float = Field(default=5.0)
    REDIS_CONNECT_TIMEOUT: float = Field(default=2.0)
    REDIS_RETRY_ON_TIMEOUT: bool = Field(default=True)
    REDIS_MAX_RETRIES: int = Field(default=3)
    REDIS_RETRY_DELAY: float = Field(default=0.1)
    
    # Celery settings
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/1")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/2")
    
    # JWT settings
    JWT_SECRET_KEY: str = Field(default="your-secret-key")
    JWT_ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)
    
    # Email settings
    MAIL_SERVER: str = Field(default="smtp.gmail.com")
    MAIL_PORT: int = Field(default=587)
    MAIL_USERNAME: str = Field(default="")
    MAIL_PASSWORD: str = Field(default="")
    MAIL_FROM: str = Field(default="noreply@example.com")
    
    # Security settings
    RATE_LIMIT_REQUESTS: int = Field(default=100)
    RATE_LIMIT_WINDOW: int = Field(default=60)  # seconds
    MAX_LOGIN_ATTEMPTS: int = Field(default=5)
    LOGIN_BLOCK_DURATION: int = Field(default=300)  # seconds
    
    # Monitoring settings
    GRAFANA_URL: str = Field(default="http://localhost:3000")
    GRAFANA_API_KEY: Optional[str] = Field(default=None)
    
    # Model settings
    ANOMALY_DETECTION_THRESHOLD: float = Field(default=-0.5)
    MODEL_UPDATE_INTERVAL: int = Field(default=86400)  # seconds
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        case_sensitive = True
        env_prefix = ""

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

# Global settings instance
settings = get_settings()
