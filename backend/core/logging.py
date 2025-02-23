"""
Security logging configuration.
"""

import logging
import json
from datetime import datetime
from typing import Any, Dict
from fastapi import Request
from prometheus_client import Counter, Histogram

# Prometheus metrics
AUTH_ATTEMPTS = Counter(
    'auth_attempts_total',
    'Total authentication attempts',
    ['status', 'endpoint']
)

AUTH_DURATION = Histogram(
    'auth_duration_seconds',
    'Authentication duration in seconds',
    ['endpoint']
)

class SecurityLogger:
    """Security event logger."""
    
    def __init__(self):
        """Initialize security logger."""
        self.logger = logging.getLogger("security")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler("security.log")
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log_auth_attempt(
        self,
        request: Request,
        success: bool,
        user_id: str = None,
        details: Dict[str, Any] = None
    ):
        """Log authentication attempt."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "auth_attempt",
            "success": success,
            "ip_address": request.client.host,
            "user_agent": request.headers.get("user-agent"),
            "endpoint": str(request.url),
            "method": request.method,
            "user_id": user_id,
            "details": details or {}
        }
        
        # Log to file
        self.logger.info(json.dumps(event))
        
        # Update Prometheus metrics
        status = "success" if success else "failure"
        AUTH_ATTEMPTS.labels(
            status=status,
            endpoint=request.url.path
        ).inc()
    
    def log_security_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        user_id: str = None,
        request: Request = None
    ):
        """Log security event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "details": details
        }
        
        if request:
            event.update({
                "ip_address": request.client.host,
                "user_agent": request.headers.get("user-agent"),
                "endpoint": str(request.url),
                "method": request.method
            })
        
        self.logger.info(json.dumps(event))
    
    def log_rate_limit(self, request: Request, remaining: int):
        """Log rate limit event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "rate_limit",
            "ip_address": request.client.host,
            "endpoint": str(request.url),
            "remaining_attempts": remaining
        }
        
        self.logger.warning(json.dumps(event))
    
    def log_brute_force(
        self,
        identifier: str,
        attempts: int,
        lockout_duration: float = None
    ):
        """Log brute force protection event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "brute_force",
            "identifier": identifier,
            "attempts": attempts,
            "lockout_duration": lockout_duration
        }
        
        self.logger.warning(json.dumps(event))

# Global security logger instance
security_logger = SecurityLogger()
