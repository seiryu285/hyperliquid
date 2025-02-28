import os
import hmac
import time
import base64
import hashlib
import logging
from typing import Dict, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class HyperLiquidAuth:
    """HyperLiquid認証管理クラス"""
    
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('HYPERLIQUID_API_KEY', '')
        self.api_secret = os.getenv('HYPERLIQUID_API_SECRET', '')
        
        if not self.api_key or not self.api_secret:
            logger.warning("API credentials not found in environment variables")
            
    def generate_signature(self, timestamp: int, method: str, path: str, body: Optional[Dict] = None) -> str:
        """
        リクエスト署名の生成
        
        Args:
            timestamp (int): UNIXタイムスタンプ
            method (str): HTTPメソッド
            path (str): APIエンドポイントパス
            body (Optional[Dict]): リクエストボディ
            
        Returns:
            str: 生成された署名
        """
        try:
            message = f"{timestamp}{method}{path}"
            if body:
                message += str(body)
            
            signature = hmac.new(
                self.api_secret.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return signature
        except Exception as e:
            logger.error(f"Error generating signature: {e}")
            raise
            
    def get_auth_headers(self, method: str, path: str, body: Optional[Dict] = None) -> Dict[str, str]:
        """
        認証ヘッダーの生成
        
        Args:
            method (str): HTTPメソッド
            path (str): APIエンドポイントパス
            body (Optional[Dict]): リクエストボディ
            
        Returns:
            Dict[str, str]: 認証ヘッダー
        """
        timestamp = int(time.time() * 1000)  # ミリ秒単位のタイムスタンプ
        signature = self.generate_signature(timestamp, method, path, body)
        
        return {
            'HL-API-KEY': self.api_key,
            'HL-SIGNATURE': signature,
            'HL-TIMESTAMP': str(timestamp)
        }
