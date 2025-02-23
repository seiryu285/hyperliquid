"""
Alert service for managing notifications and alert rules.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any
import aiohttp
import smtplib
from email.mime.text import MIMEText
from motor.motor_asyncio import AsyncIOMotorClient
import json
import logging
from prometheus_client import Counter

from core.config import settings

# Prometheus metrics
ALERTS_SENT = Counter(
    'alerts_sent_total',
    'Total number of alerts sent',
    ['type', 'channel']
)

class AlertService:
    """Service for managing alerts and notifications."""
    
    def __init__(self, db_client: AsyncIOMotorClient):
        """Initialize alert service."""
        self.db = db_client[settings.MONGO_DB]
        self.logger = logging.getLogger("alert_service")
        self._notification_cache = {}
        self._rule_cache = {}
    
    async def get_notification_channels(self) -> List[Dict[str, Any]]:
        """Get all configured notification channels."""
        channels = await self.db.notification_channels.find().to_list(None)
        return channels
    
    async def create_notification_channel(self, channel: Dict[str, Any]) -> Dict[str, Any]:
        """Create new notification channel."""
        # Validate channel configuration
        if channel["type"] not in ["email", "slack", "webhook"]:
            raise ValueError("Invalid channel type")
        
        # Test channel connection
        await self._test_channel_connection(channel)
        
        result = await self.db.notification_channels.insert_one(channel)
        channel["_id"] = result.inserted_id
        
        # Update cache
        self._notification_cache[str(result.inserted_id)] = channel
        
        return channel
    
    async def get_alert_rules(self) -> List[Dict[str, Any]]:
        """Get all configured alert rules."""
        rules = await self.db.alert_rules.find().to_list(None)
        return rules
    
    async def create_alert_rule(self, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Create new alert rule."""
        # Validate rule configuration
        if rule["event_type"] not in ["auth_failure", "rate_limit", "brute_force"]:
            raise ValueError("Invalid event type")
        
        # Validate channels exist
        channels = await self.get_notification_channels()
        channel_names = [c["name"] for c in channels]
        for channel in rule["channels"]:
            if channel not in channel_names:
                raise ValueError(f"Channel not found: {channel}")
        
        result = await self.db.alert_rules.insert_one(rule)
        rule["_id"] = result.inserted_id
        
        # Update cache
        self._rule_cache[str(result.inserted_id)] = rule
        
        return rule
    
    async def update_alert_rule(
        self,
        rule_id: str,
        rule: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update existing alert rule."""
        # Validate channels exist
        channels = await self.get_notification_channels()
        channel_names = [c["name"] for c in channels]
        for channel in rule["channels"]:
            if channel not in channel_names:
                raise ValueError(f"Channel not found: {channel}")
        
        await self.db.alert_rules.update_one(
            {"_id": rule_id},
            {"$set": rule}
        )
        
        # Update cache
        self._rule_cache[rule_id] = rule
        
        return rule
    
    async def delete_alert_rule(self, rule_id: str):
        """Delete alert rule."""
        await self.db.alert_rules.delete_one({"_id": rule_id})
        
        # Update cache
        self._rule_cache.pop(rule_id, None)
    
    async def process_security_event(
        self,
        event_type: str,
        event_data: Dict[str, Any]
    ):
        """Process security event and send notifications if needed."""
        # Get relevant rules
        rules = await self.get_alert_rules()
        matching_rules = [
            r for r in rules
            if r["event_type"] == event_type and r["enabled"]
        ]
        
        for rule in matching_rules:
            # Check if threshold is exceeded
            event_count = await self._count_events(
                event_type,
                rule["window_minutes"]
            )
            
            if event_count >= rule["threshold"]:
                # Send notifications
                await self._send_notifications(rule, event_data)
    
    async def _count_events(self, event_type: str, window_minutes: int) -> int:
        """Count events of given type within time window."""
        window_start = datetime.utcnow() - timedelta(minutes=window_minutes)
        count = await self.db.security_events.count_documents({
            "event_type": event_type,
            "timestamp": {"$gte": window_start}
        })
        return count
    
    async def _send_notifications(
        self,
        rule: Dict[str, Any],
        event_data: Dict[str, Any]
    ):
        """Send notifications through configured channels."""
        channels = await self.get_notification_channels()
        channel_map = {c["name"]: c for c in channels}
        
        for channel_name in rule["channels"]:
            channel = channel_map.get(channel_name)
            if not channel or not channel["enabled"]:
                continue
            
            try:
                if channel["type"] == "email":
                    await self._send_email_notification(channel, rule, event_data)
                elif channel["type"] == "slack":
                    await self._send_slack_notification(channel, rule, event_data)
                elif channel["type"] == "webhook":
                    await self._send_webhook_notification(channel, rule, event_data)
                
                ALERTS_SENT.labels(
                    type=rule["event_type"],
                    channel=channel["type"]
                ).inc()
            except Exception as e:
                self.logger.error(
                    f"Failed to send notification through {channel_name}: {str(e)}"
                )
    
    async def _send_email_notification(
        self,
        channel: Dict[str, Any],
        rule: Dict[str, Any],
        event_data: Dict[str, Any]
    ):
        """Send email notification."""
        msg = MIMEText(
            f"Alert: {rule['name']}\n\n"
            f"Description: {rule['description']}\n"
            f"Event Data: {json.dumps(event_data, indent=2)}"
        )
        msg["Subject"] = f"Security Alert: {rule['name']}"
        msg["From"] = settings.MAIL_FROM
        msg["To"] = channel["config"]["to_address"]
        
        with smtplib.SMTP(settings.MAIL_SERVER, settings.MAIL_PORT) as server:
            server.starttls()
            server.login(settings.MAIL_USERNAME, settings.MAIL_PASSWORD)
            server.send_message(msg)
    
    async def _send_slack_notification(
        self,
        channel: Dict[str, Any],
        rule: Dict[str, Any],
        event_data: Dict[str, Any]
    ):
        """Send Slack notification."""
        webhook_url = channel["config"]["webhook_url"]
        message = {
            "text": f"*Security Alert: {rule['name']}*\n"
                   f"Description: {rule['description']}\n"
                   f"```{json.dumps(event_data, indent=2)}```"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=message) as response:
                if response.status >= 400:
                    raise ValueError(
                        f"Slack notification failed: {await response.text()}"
                    )
    
    async def _send_webhook_notification(
        self,
        channel: Dict[str, Any],
        rule: Dict[str, Any],
        event_data: Dict[str, Any]
    ):
        """Send webhook notification."""
        webhook_url = channel["config"]["url"]
        payload = {
            "alert_name": rule["name"],
            "description": rule["description"],
            "event_type": rule["event_type"],
            "event_data": event_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status >= 400:
                    raise ValueError(
                        f"Webhook notification failed: {await response.text()}"
                    )
    
    async def _test_channel_connection(self, channel: Dict[str, Any]):
        """Test notification channel connection."""
        try:
            test_data = {
                "message": "Test notification",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if channel["type"] == "email":
                await self._send_email_notification(
                    channel,
                    {"name": "Test Alert", "description": "Test notification"},
                    test_data
                )
            elif channel["type"] == "slack":
                await self._send_slack_notification(
                    channel,
                    {"name": "Test Alert", "description": "Test notification"},
                    test_data
                )
            elif channel["type"] == "webhook":
                await self._send_webhook_notification(
                    channel,
                    {"name": "Test Alert", "description": "Test notification"},
                    test_data
                )
        except Exception as e:
            raise ValueError(f"Channel connection test failed: {str(e)}")
