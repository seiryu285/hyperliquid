"""
Security service for handling password reset and 2FA functionality.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import pyotp
import qrcode
import io
import base64
from fastapi import HTTPException, status
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from motor.motor_asyncio import AsyncIOMotorClient
import secrets

from models.user import UserInDB, TwoFactorSetup
from core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityService:
    """Service for handling security-related functionality."""
    
    def __init__(self, db_client: AsyncIOMotorClient):
        """Initialize the security service.
        
        Args:
            db_client: MongoDB client instance
        """
        self.db = db_client[settings.MONGO_DB]
        self.users_collection = self.db.users
        
        # Configure email client
        self.mail_config = ConnectionConfig(
            MAIL_USERNAME=settings.MAIL_USERNAME,
            MAIL_PASSWORD=settings.MAIL_PASSWORD,
            MAIL_FROM=settings.MAIL_FROM,
            MAIL_PORT=settings.MAIL_PORT,
            MAIL_SERVER=settings.MAIL_SERVER,
            MAIL_TLS=True,
            MAIL_SSL=False,
            USE_CREDENTIALS=True
        )
        self.mail_client = FastMail(self.mail_config)
    
    async def initiate_password_reset(self, email: str) -> bool:
        """Initiate password reset process.
        
        Args:
            email: User's email address
            
        Returns:
            bool: True if reset email was sent
        """
        user = await self.users_collection.find_one({"email": email})
        if not user:
            # Return True to prevent email enumeration
            return True
        
        user_db = UserInDB(**user)
        reset_token = user_db.generate_password_reset_token()
        
        # Update user with reset token
        await self.users_collection.update_one(
            {"_id": user_db.id},
            {
                "$set": {
                    "password_reset_token": user_db.password_reset_token,
                    "password_reset_expires": user_db.password_reset_expires
                }
            }
        )
        
        # Send reset email
        reset_url = f"{settings.FRONTEND_URL}/reset-password?token={reset_token}"
        message = MessageSchema(
            subject="Password Reset Request",
            recipients=[email],
            body=f"""
            You have requested to reset your password.
            Please click the following link to reset your password:
            {reset_url}
            
            This link will expire in 24 hours.
            If you did not request this reset, please ignore this email.
            """,
            subtype="html"
        )
        
        try:
            await self.mail_client.send_message(message)
            return True
        except Exception as e:
            logger.error(f"Failed to send reset email: {e}")
            return False
    
    async def complete_password_reset(
        self,
        token: str,
        new_password: str
    ) -> bool:
        """Complete password reset process.
        
        Args:
            token: Reset token
            new_password: New password
            
        Returns:
            bool: True if password was reset
            
        Raises:
            HTTPException: If token is invalid or expired
        """
        user = await self.users_collection.find_one({
            "password_reset_token": token,
            "password_reset_expires": {"$gt": datetime.utcnow()}
        })
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )
        
        user_db = UserInDB(**user)
        if not user_db.verify_reset_token(token):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid reset token"
            )
        
        # Update password and clear reset token
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(new_password.encode(), salt)
        
        await self.users_collection.update_one(
            {"_id": user_db.id},
            {
                "$set": {
                    "hashed_password": hashed_password.decode(),
                    "password_reset_token": None,
                    "password_reset_expires": None
                }
            }
        )
        
        return True
    
    async def setup_2fa(self, user_id: str) -> TwoFactorSetup:
        """Set up 2FA for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            TwoFactorSetup: 2FA setup information
            
        Raises:
            HTTPException: If user not found
        """
        user = await self.users_collection.find_one({"_id": user_id})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        user_db = UserInDB(**user)
        secret = user_db.generate_2fa_secret()
        
        # Generate QR code
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(
            user_db.email,
            issuer_name="HyperLiquid Trading"
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        qr_code = base64.b64encode(buffer.getvalue()).decode()
        
        # Generate backup codes
        backup_codes = [secrets.token_hex(4) for _ in range(10)]
        hashed_backup_codes = [
            bcrypt.hashpw(code.encode(), bcrypt.gensalt()).decode()
            for code in backup_codes
        ]
        
        # Update user with 2FA information
        await self.users_collection.update_one(
            {"_id": user_db.id},
            {
                "$set": {
                    "totp_secret": secret,
                    "backup_codes": hashed_backup_codes,
                    "is_2fa_enabled": True
                }
            }
        )
        
        return TwoFactorSetup(
            secret=secret,
            qr_code=qr_code,
            backup_codes=backup_codes
        )
    
    async def verify_2fa(
        self,
        user_id: str,
        token: str
    ) -> bool:
        """Verify 2FA token.
        
        Args:
            user_id: User ID
            token: TOTP token or backup code
            
        Returns:
            bool: True if token is valid
        """
        user = await self.users_collection.find_one({"_id": user_id})
        if not user:
            return False
        
        user_db = UserInDB(**user)
        
        # Try TOTP verification first
        if user_db.verify_2fa_token(token):
            return True
        
        # Check backup codes
        if "backup_codes" in user:
            for idx, hashed_code in enumerate(user["backup_codes"]):
                if bcrypt.checkpw(token.encode(), hashed_code.encode()):
                    # Remove used backup code
                    backup_codes = user["backup_codes"]
                    backup_codes.pop(idx)
                    await self.users_collection.update_one(
                        {"_id": user_id},
                        {"$set": {"backup_codes": backup_codes}}
                    )
                    return True
        
        return False
    
    async def disable_2fa(self, user_id: str, token: str) -> bool:
        """Disable 2FA for a user.
        
        Args:
            user_id: User ID
            token: Current TOTP token for verification
            
        Returns:
            bool: True if 2FA was disabled
            
        Raises:
            HTTPException: If token is invalid
        """
        if not await self.verify_2fa(user_id, token):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid 2FA token"
            )
        
        await self.users_collection.update_one(
            {"_id": user_id},
            {
                "$set": {
                    "is_2fa_enabled": False,
                    "totp_secret": None,
                    "backup_codes": []
                }
            }
        )
        
        return True
