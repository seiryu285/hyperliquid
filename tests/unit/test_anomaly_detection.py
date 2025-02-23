"""
Unit tests for anomaly detection system.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from ml.anomaly_detection import AnomalyDetector

def test_anomaly_detection(
    anomaly_detector: AnomalyDetector,
    test_data: dict
):
    """Test anomaly detection functionality."""
    # Generate normal data
    normal_data = []
    base_time = datetime.utcnow()
    
    for i in range(100):
        data = test_data.copy()
        data["timestamp"] = (base_time + timedelta(minutes=i)).isoformat()
        data["price"] = 50000.0 + np.random.normal(0, 100)
        data["volume"] = 100.0 + np.random.normal(0, 10)
        normal_data.append(data)
    
    # Train model with normal data
    anomaly_detector.train(normal_data)
    
    # Test normal data point
    normal_point = test_data.copy()
    normal_point["price"] = 50100.0
    normal_point["volume"] = 105.0
    
    result = anomaly_detector.detect_anomalies([normal_point])
    assert len(result) == 1
    assert result[0]["is_anomaly"] is False
    
    # Test anomalous data point
    anomaly_point = test_data.copy()
    anomaly_point["price"] = 60000.0  # Significant price jump
    anomaly_point["volume"] = 1000.0  # Unusual volume
    
    result = anomaly_detector.detect_anomalies([anomaly_point])
    assert len(result) == 1
    assert result[0]["is_anomaly"] is True
    assert result[0]["score"] < -0.5  # Default threshold

def test_batch_anomaly_detection(
    anomaly_detector: AnomalyDetector,
    test_data: dict
):
    """Test batch anomaly detection."""
    # Generate mixed data
    mixed_data = []
    base_time = datetime.utcnow()
    
    for i in range(10):
        # Normal data
        normal = test_data.copy()
        normal["timestamp"] = (base_time + timedelta(minutes=i)).isoformat()
        normal["price"] = 50000.0 + np.random.normal(0, 100)
        normal["volume"] = 100.0 + np.random.normal(0, 10)
        mixed_data.append(normal)
        
        # Anomalous data
        if i % 3 == 0:
            anomaly = test_data.copy()
            anomaly["timestamp"] = (base_time + timedelta(minutes=i, seconds=30)).isoformat()
            anomaly["price"] = 55000.0 + np.random.normal(0, 100)
            anomaly["volume"] = 500.0 + np.random.normal(0, 50)
            mixed_data.append(anomaly)
    
    # Train model
    anomaly_detector.train(mixed_data[:5])
    
    # Test batch detection
    results = anomaly_detector.detect_anomalies(mixed_data[5:])
    
    assert len(results) == len(mixed_data[5:])
    assert any(r["is_anomaly"] for r in results)
    assert any(not r["is_anomaly"] for r in results)

def test_model_persistence(
    anomaly_detector: AnomalyDetector,
    test_data: dict
):
    """Test model saving and loading."""
    # Generate and train with data
    train_data = []
    base_time = datetime.utcnow()
    
    for i in range(50):
        data = test_data.copy()
        data["timestamp"] = (base_time + timedelta(minutes=i)).isoformat()
        data["price"] = 50000.0 + np.random.normal(0, 100)
        data["volume"] = 100.0 + np.random.normal(0, 10)
        train_data.append(data)
    
    # Train and save model
    anomaly_detector.train(train_data)
    anomaly_detector.save_model()
    
    # Create new detector instance
    new_detector = AnomalyDetector(
        db=anomaly_detector.db,
        model_path=anomaly_detector.model_path
    )
    
    # Load model and test
    new_detector.load_model()
    
    test_point = test_data.copy()
    test_point["price"] = 60000.0
    test_point["volume"] = 1000.0
    
    result1 = anomaly_detector.detect_anomalies([test_point])
    result2 = new_detector.detect_anomalies([test_point])
    
    assert result1[0]["is_anomaly"] == result2[0]["is_anomaly"]
    assert abs(result1[0]["score"] - result2[0]["score"]) < 1e-6
