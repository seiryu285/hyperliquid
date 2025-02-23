"""
Alert configuration API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
from pydantic import BaseModel, HttpUrl

from core.dependencies import get_current_user
from models.user import User
from services.alert_service import AlertService

router = APIRouter(prefix="/alert-config", tags=["alerts"])

class NotificationChannel(BaseModel):
    """Notification channel configuration."""
    type: str  # email, slack, webhook
    name: str
    config: dict
    enabled: bool = True

class AlertRule(BaseModel):
    """Alert rule configuration."""
    name: str
    description: str
    event_type: str  # auth_failure, rate_limit, brute_force
    threshold: int
    window_minutes: int
    channels: List[str]  # List of channel names
    enabled: bool = True

@router.get("/channels", response_model=List[NotificationChannel])
async def get_channels(
    current_user: User = Depends(get_current_user),
    alert_service: AlertService = Depends()
):
    """Get configured notification channels."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return await alert_service.get_notification_channels()

@router.post("/channels", response_model=NotificationChannel)
async def create_channel(
    channel: NotificationChannel,
    current_user: User = Depends(get_current_user),
    alert_service: AlertService = Depends()
):
    """Create new notification channel."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return await alert_service.create_notification_channel(channel)

@router.get("/rules", response_model=List[AlertRule])
async def get_rules(
    current_user: User = Depends(get_current_user),
    alert_service: AlertService = Depends()
):
    """Get configured alert rules."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return await alert_service.get_alert_rules()

@router.post("/rules", response_model=AlertRule)
async def create_rule(
    rule: AlertRule,
    current_user: User = Depends(get_current_user),
    alert_service: AlertService = Depends()
):
    """Create new alert rule."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return await alert_service.create_alert_rule(rule)

@router.put("/rules/{rule_id}", response_model=AlertRule)
async def update_rule(
    rule_id: str,
    rule: AlertRule,
    current_user: User = Depends(get_current_user),
    alert_service: AlertService = Depends()
):
    """Update existing alert rule."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return await alert_service.update_alert_rule(rule_id, rule)

@router.delete("/rules/{rule_id}")
async def delete_rule(
    rule_id: str,
    current_user: User = Depends(get_current_user),
    alert_service: AlertService = Depends()
):
    """Delete alert rule."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    await alert_service.delete_alert_rule(rule_id)
    return {"status": "success"}
