"""
Caching mechanism for frequently accessed data and computation results.
Provides Redis-based caching with automatic fallback to in-memory cache.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List, Union, Tuple
import redis
from redis.connection import ConnectionPool
from redis.exceptions import RedisError
import json
import logging
from functools import wraps
import time
from threading import Lock

from core.config import settings

logger = logging.getLogger(__name__)

class CacheEntry:
    """Cache entry with value and expiration time."""
    
    def __init__(self, value: Any, expire_seconds: int):
        """Initialize cache entry."""
        self.value = value
        self.expires_at = datetime.now() + timedelta(seconds=expire_seconds)
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return datetime.now() > self.expires_at

class LocalCache:
    """Thread-safe in-memory cache implementation."""
    
    def __init__(self):
        """Initialize local cache."""
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from local cache."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if entry.is_expired():
                del self._cache[key]
                return None
            return entry.value
    
    def set(self, key: str, value: Any, expire_seconds: int) -> None:
        """Set value in local cache."""
        with self._lock:
            self._cache[key] = CacheEntry(value, expire_seconds)
    
    def delete(self, key: str) -> None:
        """Delete value from local cache."""
        with self._lock:
            self._cache.pop(key, None)
    
    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()

class Cache:
    """Redis-based caching system with automatic fallback to in-memory cache."""
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        max_retries: int = 3,
        retry_delay: float = 0.1
    ):
        """Initialize cache with Redis connection and in-memory fallback.
        
        Args:
            redis_client: Optional Redis client instance
            max_retries: Maximum number of retries for Redis operations
            retry_delay: Delay between retries in seconds
        """
        self._pool = None
        if redis_client is None:
            try:
                self._pool = ConnectionPool.from_url(
                    settings.REDIS_URL,
                    max_connections=settings.REDIS_MAX_CONNECTIONS,
                    socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
                    socket_connect_timeout=settings.REDIS_CONNECT_TIMEOUT
                )
                self.redis = redis.Redis(connection_pool=self._pool)
                logger.info("Successfully connected to Redis")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.redis = None
        else:
            self.redis = redis_client
        
        self._local_cache = LocalCache()
        self._max_retries = max_retries
        self._retry_delay = retry_delay
    
    def _execute_with_retry(
        self,
        operation: str,
        func: callable,
        *args,
        **kwargs
    ) -> Tuple[Any, bool]:
        """Execute Redis operation with retry logic.
        
        Returns:
            Tuple of (result, used_local_cache)
        """
        for attempt in range(self._max_retries):
            try:
                if self.redis is not None:
                    result = func(*args, **kwargs)
                    return result, False
            except RedisError as e:
                logger.warning(
                    f"Redis {operation} failed (attempt {attempt + 1}): {e}"
                )
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay)
                continue
            except Exception as e:
                logger.error(f"Unexpected error in {operation}: {e}")
                break
        
        logger.info(f"Falling back to local cache for {operation}")
        return None, True
    
    def get(
        self,
        key: str,
        prefix: str = "",
        default: Any = None
    ) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            prefix: Optional key prefix
            default: Default value if key not found
        """
        full_key = f"{prefix}:{key}" if prefix else key
        
        # Try Redis first
        result, use_local = self._execute_with_retry(
            "get",
            self.redis.get,
            full_key
        )
        
        if not use_local and result is not None:
            try:
                return json.loads(result)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode cache value: {e}")
                return default
        
        # Fallback to local cache
        value = self._local_cache.get(full_key)
        if value is not None:
            return value
        
        return default
    
    def set(
        self,
        key: str,
        value: Any,
        expire_seconds: int = 300,
        prefix: str = ""
    ) -> bool:
        """Set value in cache with expiration.
        
        Args:
            key: Cache key
            value: Value to store
            expire_seconds: Expiration time in seconds
            prefix: Optional key prefix
        """
        try:
            full_key = f"{prefix}:{key}" if prefix else key
            json_value = json.dumps(value)
            
            # Try Redis first
            _, use_local = self._execute_with_retry(
                "set",
                self.redis.setex,
                full_key,
                expire_seconds,
                json_value
            )
            
            # Always update local cache
            self._local_cache.set(full_key, value, expire_seconds)
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache key {full_key}: {e}")
            return False
    
    def delete(self, key: str, prefix: str = "") -> bool:
        """Delete value from cache.
        
        Args:
            key: Cache key
            prefix: Optional key prefix
        """
        try:
            full_key = f"{prefix}:{key}" if prefix else key
            
            # Try Redis first
            self._execute_with_retry("delete", self.redis.delete, full_key)
            
            # Always delete from local cache
            self._local_cache.delete(full_key)
            return True
            
        except Exception as e:
            logger.error(f"Error deleting cache key {full_key}: {e}")
            return False
    
    def exists(self, key: str, prefix: str = "") -> bool:
        """Check if key exists in cache.
        
        Args:
            key: Cache key
            prefix: Optional key prefix
        """
        try:
            full_key = f"{prefix}:{key}" if prefix else key
            
            # Try Redis first
            result, use_local = self._execute_with_retry(
                "exists",
                self.redis.exists,
                full_key
            )
            
            if not use_local:
                return bool(result)
            
            # Check local cache
            return self._local_cache.get(full_key) is not None
            
        except Exception as e:
            logger.error(f"Error checking cache key {full_key}: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            # Try Redis first
            self._execute_with_retry("clear", self.redis.flushdb)
            
            # Always clear local cache
            self._local_cache.clear()
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def bulk_get(
        self,
        keys: List[str],
        prefix: str = "",
        default: Any = None
    ) -> Dict[str, Any]:
        """Get multiple values from cache using pipeline.
        
        Args:
            keys: List of cache keys
            prefix: Optional key prefix
            default: Default value for missing keys
        """
        result = {}
        full_keys = [f"{prefix}:{k}" if prefix else k for k in keys]
        
        # Try Redis pipeline first
        if self.redis is not None:
            try:
                with self.redis.pipeline() as pipe:
                    for key in full_keys:
                        pipe.get(key)
                    values = pipe.execute()
                    
                    for key, value in zip(keys, values):
                        if value is not None:
                            try:
                                result[key] = json.loads(value)
                            except json.JSONDecodeError:
                                result[key] = default
                        else:
                            result[key] = default
                    
                    return result
                    
            except RedisError as e:
                logger.warning(f"Redis bulk_get failed: {e}")
        
        # Fallback to local cache
        for key, full_key in zip(keys, full_keys):
            value = self._local_cache.get(full_key)
            result[key] = value if value is not None else default
        
        return result
    
    def bulk_set(
        self,
        items: Dict[str, Any],
        expire_seconds: int = 300,
        prefix: str = ""
    ) -> bool:
        """Set multiple values in cache using pipeline.
        
        Args:
            items: Dictionary of key-value pairs
            expire_seconds: Expiration time in seconds
            prefix: Optional key prefix
        """
        try:
            # Try Redis pipeline first
            if self.redis is not None:
                try:
                    with self.redis.pipeline() as pipe:
                        for key, value in items.items():
                            full_key = f"{prefix}:{key}" if prefix else key
                            json_value = json.dumps(value)
                            pipe.setex(full_key, expire_seconds, json_value)
                        pipe.execute()
                except RedisError as e:
                    logger.warning(f"Redis bulk_set failed: {e}")
            
            # Always update local cache
            for key, value in items.items():
                full_key = f"{prefix}:{key}" if prefix else key
                self._local_cache.set(full_key, value, expire_seconds)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in bulk_set: {e}")
            return False
    
    def bulk_delete(self, keys: List[str], prefix: str = "") -> bool:
        """Delete multiple values from cache using pipeline.
        
        Args:
            keys: List of cache keys
            prefix: Optional key prefix
        """
        try:
            full_keys = [f"{prefix}:{k}" if prefix else k for k in keys]
            
            # Try Redis pipeline first
            if self.redis is not None:
                try:
                    with self.redis.pipeline() as pipe:
                        for key in full_keys:
                            pipe.delete(key)
                        pipe.execute()
                except RedisError as e:
                    logger.warning(f"Redis bulk_delete failed: {e}")
            
            # Always delete from local cache
            for key in full_keys:
                self._local_cache.delete(key)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in bulk_delete: {e}")
            return False

# Global cache instance
cache = Cache()
