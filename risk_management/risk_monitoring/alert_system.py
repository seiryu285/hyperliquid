"""
Alert Notification System for HyperLiquid Trading

This module implements a robust notification system that sends alerts through various
channels (email, Slack, webhook) when risk thresholds are exceeded. It includes
retry logic and rate limiting to prevent alert fatigue.
"""

import smtplib
import requests
import json
import yaml
import logging
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from threading import Lock
from prometheus_client import Counter, Histogram, Gauge

# Import metrics collector
from monitoring.metrics_collector import MetricsCollector, MetricsConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alert_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AlertConfig:
    """Configuration for alert channels."""
    email: Dict[str, str]
    slack_webhook: str
    custom_webhook: str
    enabled_channels: List[str]

@dataclass
class AlertMessage:
    """Container for alert message content."""
    severity: str
    title: str
    description: str
    metrics: Dict[str, float]
    timestamp: float

class AlertSystem:
    """Main class for handling alert notifications."""
    
    def __init__(
        self,
        config_path: str = "config/api_keys.yaml",
        env_config_path: str = "config/environments/development.yaml"
    ):
        """Initialize the alert system with configuration parameters.
        
        Args:
            config_path: Path to the API keys configuration file
            env_config_path: Path to the environment configuration file
        """
        self.alert_config = self._load_config(config_path)
        self.env_config = self._load_config(env_config_path)
        
        # Initialize rate limiting
        self.last_alert_time: Dict[str, float] = {}
        self.alert_lock = Lock()
        
        # Alert cooldown period (in seconds)
        self.cooldown_period = self.env_config['monitoring']['alert_cooldown']
        
        # Initialize metrics collector
        metrics_config = MetricsConfig(port=8002)
        self.metrics_collector = MetricsCollector(metrics_config)
        self.metrics_collector.start()
        
        # Initialize Prometheus metrics
        self.alerts_total = Counter(
            'alerts_total',
            'Total number of alerts sent',
            ['severity', 'channel']
        )
        self.alert_send_duration = Histogram(
            'alert_send_duration_seconds',
            'Time spent sending alerts',
            ['channel']
        )
        self.alert_errors = Counter(
            'alert_errors_total',
            'Number of alert sending errors',
            ['channel']
        )
        self.alert_rate = Gauge(
            'alert_rate',
            'Number of alerts per minute',
            ['severity']
        )
        
        logger.info("Alert System initialized with configuration from %s", config_path)

    def __del__(self):
        """Cleanup when the object is destroyed."""
        if hasattr(self, 'metrics_collector'):
            self.metrics_collector.stop()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dict containing configuration parameters
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise

    def _can_send_alert(self, alert_type: str) -> bool:
        """Check if enough time has passed since the last alert of this type.
        
        Args:
            alert_type: Type of alert to check
            
        Returns:
            bool: True if alert can be sent, False otherwise
        """
        with self.alert_lock:
            current_time = time.time()
            last_time = self.last_alert_time.get(alert_type, 0)
            
            if current_time - last_time >= self.cooldown_period:
                self.last_alert_time[alert_type] = current_time
                return True
            return False

    def send_email_alert(
        self,
        message: AlertMessage,
        retry_count: int = 3
    ) -> bool:
        """Send an alert via email.
        
        Args:
            message: Alert message to send
            retry_count: Number of retry attempts
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._can_send_alert('email'):
            logger.info("Email alert skipped due to cooldown period")
            return False
            
        email_config = self.alert_config['email']
        
        msg = MIMEMultipart()
        msg['From'] = email_config['username']
        msg['To'] = email_config.get('alert_recipients', email_config['username'])
        msg['Subject'] = f"[{message.severity}] HyperLiquid Trading Alert: {message.title}"
        
        # Create HTML content
        html_content = f"""
        <html>
            <body>
                <h2>{message.title}</h2>
                <p><strong>Severity:</strong> {message.severity}</p>
                <p><strong>Time:</strong> {datetime.fromtimestamp(message.timestamp)}</p>
                <p>{message.description}</p>
                <h3>Risk Metrics:</h3>
                <ul>
                    {''.join(f'<li><strong>{k}:</strong> {v:.4f}</li>' for k, v in message.metrics.items())}
                </ul>
            </body>
        </html>
        """
        
        msg.attach(MIMEText(html_content, 'html'))
        
        for attempt in range(retry_count):
            try:
                with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                    server.starttls()
                    server.login(email_config['username'], email_config['password'])
                    server.send_message(msg)
                logger.info("Email alert sent successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to send email alert (attempt {attempt + 1}): {e}")
                if attempt == retry_count - 1:
                    return False
                time.sleep(2 ** attempt)  # Exponential backoff

    def send_slack_alert(
        self,
        message: AlertMessage,
        retry_count: int = 3
    ) -> bool:
        """Send an alert via Slack webhook.
        
        Args:
            message: Alert message to send
            retry_count: Number of retry attempts
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._can_send_alert('slack'):
            logger.info("Slack alert skipped due to cooldown period")
            return False
            
        webhook_url = self.alert_config['slack_webhook']
        
        # Create Slack message
        slack_message = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"🚨 {message.title}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Severity:*\n{message.severity}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Time:*\n{datetime.fromtimestamp(message.timestamp)}"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": message.description
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Risk Metrics:*\n" + "\n".join(f"• {k}: {v:.4f}" for k, v in message.metrics.items())
                    }
                }
            ]
        }
        
        for attempt in range(retry_count):
            try:
                response = requests.post(
                    webhook_url,
                    json=slack_message,
                    timeout=5
                )
                response.raise_for_status()
                logger.info("Slack alert sent successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to send Slack alert (attempt {attempt + 1}): {e}")
                if attempt == retry_count - 1:
                    return False
                time.sleep(2 ** attempt)  # Exponential backoff

    def send_webhook_alert(
        self,
        message: AlertMessage,
        retry_count: int = 3
    ) -> bool:
        """Send an alert via custom webhook.
        
        Args:
            message: Alert message to send
            retry_count: Number of retry attempts
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._can_send_alert('webhook'):
            logger.info("Webhook alert skipped due to cooldown period")
            return False
            
        webhook_url = self.alert_config['custom_webhook']
        
        payload = {
            'timestamp': message.timestamp,
            'severity': message.severity,
            'title': message.title,
            'description': message.description,
            'metrics': message.metrics
        }
        
        for attempt in range(retry_count):
            try:
                response = requests.post(
                    webhook_url,
                    json=payload,
                    timeout=5
                )
                response.raise_for_status()
                logger.info("Webhook alert sent successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to send webhook alert (attempt {attempt + 1}): {e}")
                if attempt == retry_count - 1:
                    return False
                time.sleep(2 ** attempt)  # Exponential backoff

    def send_alert(
        self,
        message: AlertMessage,
        channels: List[str] = None
    ) -> bool:
        """Send an alert through specified channels.
        
        Args:
            message: Alert message to send
            channels: List of channels to use (email, slack, webhook)
            
        Returns:
            bool: True if alert was sent successfully through any channel
        """
        if channels is None:
            channels = self.alert_config['enabled_channels']
        
        success = False
        start_time = time.time()
        
        try:
            for channel in channels:
                if not self._can_send_alert(channel):
                    logger.info(f"{channel} alert skipped due to cooldown period")
                    continue
                
                channel_start_time = time.time()
                
                try:
                    if channel == 'email':
                        success |= self.send_email_alert(message)
                    elif channel == 'slack':
                        success |= self.send_slack_alert(message)
                    elif channel == 'webhook':
                        success |= self.send_webhook_alert(message)
                    
                    if success:
                        self.alerts_total.labels(
                            severity=message.severity,
                            channel=channel
                        ).inc()
                        
                        # Record alert send duration
                        self.alert_send_duration.labels(channel=channel).observe(
                            time.time() - channel_start_time
                        )
                        
                except Exception as e:
                    logger.error(f"Failed to send {channel} alert: {e}")
                    self.alert_errors.labels(channel=channel).inc()
            
            # Update alert rate
            self.alert_rate.labels(severity=message.severity).set(
                60 / (time.time() - self.last_alert_time.get(message.severity, 0))
                if message.severity in self.last_alert_time
                else 0
            )
            
            return success
            
        finally:
            # Record metrics
            self.metrics_collector.record_request_duration(time.time() - start_time)

if __name__ == '__main__':
    print('Risk Alert System')
    # Example usage
    alert_system = AlertSystem()
    test_message = AlertMessage(
        severity="HIGH",
        title="Margin Buffer Below Threshold",
        description="Current margin buffer ratio has fallen below the minimum threshold of 1.5",
        metrics={'margin_buffer_ratio': 1.2, 'liquidation_risk': 0.8},
        timestamp=time.time()
    )
    alert_system.send_alert(test_message)