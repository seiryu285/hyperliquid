"""
Integration tests for the monitoring and alerting system.

This module tests the integration between the market data streaming,
alert system, and Prometheus metrics.
"""

import asyncio
import logging
import os
import pytest
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from market_data.stream_manager import StreamManager
from market_data.websocket_client import WebSocketClient
from monitoring.alerts.alert_manager import AlertManager
from risk_management.risk_monitoring.alert_system import alert_system, AlertType, AlertLevel

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@pytest.fixture
async def alert_manager():
    """Set up and tear down the alert manager."""
    manager = AlertManager()
    yield manager

@pytest.fixture
async def stream_manager():
    """Set up and tear down the stream manager."""
    manager = StreamManager("config/market_data.yaml")
    yield manager
    await manager.stop()

@pytest.mark.asyncio
async def test_websocket_metrics(stream_manager):
    """Test that WebSocket metrics are properly initialized and updated."""
    # Start the stream manager
    await stream_manager.start()
    
    # Check that the WebSocket client metrics are initialized
    assert stream_manager.ws_client.ws_connected is not None
    assert stream_manager.ws_client.ws_messages_received is not None
    
    # Wait for some activity
    await asyncio.sleep(5)
    
    # Stop the stream manager
    await stream_manager.stop()

@pytest.mark.asyncio
async def test_alert_system_integration(alert_manager):
    """Test that the alert system properly integrates with the alert manager."""
    # Trigger a test alert
    test_alert = alert_system.trigger_alert(
        alert_type=AlertType.CUSTOM,
        level=AlertLevel.INFO,
        message="Test alert for integration testing",
        data={"test": True, "timestamp": datetime.utcnow().isoformat()}
    )
    
    # Check that the alert was created
    assert test_alert is not None
    assert test_alert.message == "Test alert for integration testing"
    
    # Get recent alerts
    recent_alerts = alert_system.get_recent_alerts(limit=1)
    assert len(recent_alerts) > 0
    assert recent_alerts[0]["message"] == "Test alert for integration testing"

@pytest.mark.asyncio
async def test_error_handling(stream_manager):
    """Test that errors in the stream manager are properly handled and alerted."""
    # Start the stream manager
    await stream_manager.start()
    
    # Simulate errors by directly incrementing the error count
    stream_manager.error_count = stream_manager.max_errors - 1
    
    # Trigger one more error to exceed the threshold
    try:
        # Force an error in the message processing
        await stream_manager.validator.validate_message({"invalid": "message"})
    except Exception:
        # Expected to fail
        pass
    
    # Wait for the alert to be triggered
    await asyncio.sleep(1)
    
    # Check that an alert was triggered
    recent_alerts = alert_system.get_recent_alerts(
        limit=5, 
        alert_type=AlertType.SYSTEM_ERROR
    )
    
    # There should be at least one system error alert
    assert any(alert["alert_type"] == AlertType.SYSTEM_ERROR.value for alert in recent_alerts)
    
    # Stop the stream manager
    await stream_manager.stop()

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
