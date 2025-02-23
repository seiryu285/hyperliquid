"""
Test configuration and fixtures.
"""

import pytest
from typing import Generator
import mongomock
import redis
from datetime import datetime
from unittest.mock import MagicMock
from pathlib import Path
import os

# Set environment variables for testing
os.environ["REDIS_URL"] = "redis://localhost:6379"
os.environ["REDIS_MAX_CONNECTIONS"] = "10"
os.environ["REDIS_SOCKET_TIMEOUT"] = "5.0"
os.environ["REDIS_CONNECT_TIMEOUT"] = "2.0"
os.environ["REDIS_RETRY_ON_TIMEOUT"] = "true"
os.environ["REDIS_MAX_RETRIES"] = "3"
os.environ["REDIS_RETRY_DELAY"] = "0.1"

from core.cache import Cache, LocalCache, CacheEntry
from ml.anomaly_detection import AnomalyDetector

@pytest.fixture
def mock_mongodb() -> Generator[mongomock.MongoClient, None, None]:
    """Create mock MongoDB client."""
    client = mongomock.MongoClient()
    yield client
    client.close()

@pytest.fixture
def mock_redis() -> Generator[redis.Redis, None, None]:
    """Create mock Redis client."""
    mock_redis = MagicMock(spec=redis.Redis)
    yield mock_redis

@pytest.fixture
def cache(mock_redis: redis.Redis) -> Cache:
    """Create Cache instance with mock Redis."""
    return Cache(mock_redis)

@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    """Create temporary path for model files."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    return model_dir / "anomaly_detector.joblib"

@pytest.fixture
def anomaly_detector(mock_mongodb: mongomock.MongoClient, model_path: Path) -> AnomalyDetector:
    """Create AnomalyDetector instance with mock database."""
    return AnomalyDetector(
        db=mock_mongodb["hyperliquid"],
        model_path=model_path
    )

@pytest.fixture
def test_data() -> dict:
    """Create test market data."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": "BTC-USD",
        "price": 50000.0,
        "volume": 100.0,
        "side": "buy",
        "metrics": {
            "volatility": 0.02,
            "spread": 0.001,
            "depth": 1000.0
        }
    }

@pytest.fixture
def test_alert_template() -> dict:
    """Create test alert template."""
    return {
        "name": "Test Alert",
        "type": "anomaly",
        "channels": ["email", "slack"],
        "template": {
            "email": "Alert: {{ alert.name }}\nScore: {{ alert.score }}",
            "slack": "*Alert*: {{ alert.name }}\n*Score*: {{ alert.score }}"
        }
    }
