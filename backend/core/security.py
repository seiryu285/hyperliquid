"""
Security utilities and middleware.
"""

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import redis
from redis.exceptions import ConnectionError
import json

from core.config import settings

class RateLimiter:
    """Rate limiter using Redis."""
    
    def __init__(self):
        """Initialize rate limiter."""
        try:
            self.redis = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True
            )
        except ConnectionError:
            # Fallback to in-memory storage if Redis is not available
            self.redis = None
            self._storage: Dict[str, Dict] = {}
    
    def _get_key(self, request: Request) -> str:
        """Get rate limit key based on IP and endpoint."""
        return f"rate_limit:{request.client.host}:{request.url.path}"
    
    async def is_rate_limited(
        self,
        request: Request,
        limit: int = 100,
        window: int = 60
    ) -> Tuple[bool, Optional[int]]:
        """Check if request is rate limited."""
        key = self._get_key(request)
        now = int(time.time())
        
        if self.redis:
            try:
                pipe = self.redis.pipeline()
                pipe.zremrangebyscore(key, 0, now - window)
                pipe.zadd(key, {str(now): now})
                pipe.zcard(key)
                pipe.expire(key, window)
                _, _, count, _ = pipe.execute()
            except ConnectionError:
                return False, None
        else:
            # In-memory fallback
            if key not in self._storage:
                self._storage[key] = []
            
            self._storage[key] = [
                ts for ts in self._storage[key]
                if ts > now - window
            ]
            self._storage[key].append(now)
            count = len(self._storage[key])
        
        return count > limit, limit - count

class BruteForceProtection:
    """Protection against brute force attacks."""
    
    def __init__(self):
        """Initialize brute force protection."""
        try:
            self.redis = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True
            )
        except ConnectionError:
            self.redis = None
            self._storage: Dict[str, Dict] = {}
    
    def _get_key(self, identifier: str) -> str:
        """Get key for storing attempt data."""
        return f"login_attempts:{identifier}"
    
    async def record_attempt(
        self,
        identifier: str,
        success: bool
    ) -> Tuple[bool, Optional[int]]:
        """Record login attempt and check if account should be locked."""
        key = self._get_key(identifier)
        now = datetime.utcnow()
        attempt = json.dumps({
            "timestamp": now.timestamp(),
            "success": success
        })
        
        if self.redis:
            try:
                pipe = self.redis.pipeline()
                pipe.lpush(key, attempt)
                pipe.ltrim(key, 0, settings.MAX_LOGIN_ATTEMPTS - 1)
                pipe.expire(key, settings.ACCOUNT_LOCKOUT_MINUTES * 60)
                pipe.lrange(key, 0, -1)
                _, _, _, attempts = pipe.execute()
                attempts = [json.loads(a) for a in attempts]
            except ConnectionError:
                return False, None
        else:
            # In-memory fallback
            if key not in self._storage:
                self._storage[key] = []
            
            self._storage[key].append({
                "timestamp": now.timestamp(),
                "success": success
            })
            
            if len(self._storage[key]) > settings.MAX_LOGIN_ATTEMPTS:
                self._storage[key].pop()
            
            attempts = self._storage[key]
        
        # Check recent failed attempts
        recent_failures = [
            a for a in attempts
            if not a["success"] and
            now - datetime.fromtimestamp(a["timestamp"]) <
            timedelta(minutes=settings.ACCOUNT_LOCKOUT_MINUTES)
        ]
        
        should_lock = len(recent_failures) >= settings.MAX_LOGIN_ATTEMPTS
        remaining_attempts = settings.MAX_LOGIN_ATTEMPTS - len(recent_failures)
        
        return should_lock, remaining_attempts

    async def is_locked(self, identifier: str) -> Tuple[bool, Optional[float]]:
        """Check if account is locked and get remaining lockout time."""
        key = self._get_key(identifier)
        now = datetime.utcnow()
        
        if self.redis:
            try:
                attempts = self.redis.lrange(key, 0, -1)
                attempts = [json.loads(a) for a in attempts]
            except ConnectionError:
                return False, None
        else:
            attempts = self._storage.get(key, [])
        
        recent_failures = [
            a for a in attempts
            if not a["success"] and
            now - datetime.fromtimestamp(a["timestamp"]) <
            timedelta(minutes=settings.ACCOUNT_LOCKOUT_MINUTES)
        ]
        
        if len(recent_failures) >= settings.MAX_LOGIN_ATTEMPTS:
            newest_failure = max(
                recent_failures,
                key=lambda x: x["timestamp"]
            )
            lockout_end = datetime.fromtimestamp(newest_failure["timestamp"]) + \
                timedelta(minutes=settings.ACCOUNT_LOCKOUT_MINUTES)
            remaining = (lockout_end - now).total_seconds()
            return True, remaining if remaining > 0 else None
        
        return False, None

class SecurityBearer(HTTPBearer):
    """Enhanced security bearer with rate limiting."""
    
    def __init__(self):
        """Initialize security bearer."""
        super().__init__(auto_error=True)
        self.rate_limiter = RateLimiter()
    
    async def __call__(self, request: Request):
        """Process request with rate limiting."""
        is_limited, remaining = await self.rate_limiter.is_rate_limited(request)
        if is_limited:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again in {remaining} seconds."
            )
        
        return await super().__call__(request)
