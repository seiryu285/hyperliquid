"""
User models for authentication and authorization.
"""

from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field, validator
import secrets
from passlib.context import CryptContext
import pyotp

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserBase(BaseModel):
    email: EmailStr
    username: str
    is_active: bool = True
    is_2fa_enabled: bool = False

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)
    
    @validator('password')
    def validate_password(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one number')
        if not any(c in '!@#$%^&*()' for c in v):
            raise ValueError('Password must contain at least one special character')
        return v

class User(UserBase):
    id: str
    created_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True

class UserInDB(User):
    hashed_password: str
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    totp_secret: Optional[str] = None
    backup_codes: List[str] = []
    password_reset_token: Optional[str] = None
    password_reset_expires: Optional[datetime] = None

    @classmethod
    def from_user_create(cls, user_create: UserCreate, **kwargs):
        """Create a UserInDB instance from UserCreate model."""
        values = user_create.model_dump()
        values.pop("password")
        values["hashed_password"] = pwd_context.hash(user_create.password)
        values["created_at"] = datetime.utcnow()
        values.update(kwargs)
        return cls(**values)

    def verify_password(self, password: str) -> bool:
        """Verify password against hash."""
        return pwd_context.verify(password, self.hashed_password)

    def generate_2fa_secret(self) -> str:
        """Generate new TOTP secret."""
        self.totp_secret = pyotp.random_base32()
        return self.totp_secret

    def verify_2fa_token(self, token: str) -> bool:
        """Verify TOTP token."""
        if not self.totp_secret:
            return False
        totp = pyotp.TOTP(self.totp_secret)
        return totp.verify(token)

    def generate_password_reset_token(self) -> str:
        """Generate password reset token."""
        self.password_reset_token = secrets.token_urlsafe(32)
        self.password_reset_expires = datetime.utcnow() + timedelta(hours=24)
        return self.password_reset_token

    def verify_reset_token(self, token: str) -> bool:
        """Verify password reset token."""
        if not self.password_reset_token or not self.password_reset_expires:
            return False
        if datetime.utcnow() > self.password_reset_expires:
            return False
        return secrets.compare_digest(self.password_reset_token, token)

class PasswordReset(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8)
    
    @validator('new_password')
    def validate_password(cls, v):
        """Validate password strength."""
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one number')
        if not any(c in '!@#$%^&*()' for c in v):
            raise ValueError('Password must contain at least one special character')
        return v

class TwoFactorSetup(BaseModel):
    secret: str
    qr_code: str
    backup_codes: List[str]

class TwoFactorVerify(BaseModel):
    token: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
