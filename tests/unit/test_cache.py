"""
Unit tests for caching system.
"""

import pytest
from datetime import datetime, timedelta
import json

from core.cache import Cache

def test_cache_set_get(cache: Cache):
    """Test basic cache set and get operations."""
    # Test with string
    cache.set("test_str", "value")
    assert cache.get("test_str") == "value"
    
    # Test with integer
    cache.set("test_int", 42)
    assert cache.get("test_int") == 42
    
    # Test with float
    cache.set("test_float", 3.14)
    assert cache.get("test_float") == 3.14
    
    # Test with dictionary
    data = {"key": "value", "number": 42}
    cache.set("test_dict", data)
    assert cache.get("test_dict") == data
    
    # Test with list
    data = [1, 2, 3, "test"]
    cache.set("test_list", data)
    assert cache.get("test_list") == data

def test_cache_expiration(cache: Cache):
    """Test cache expiration."""
    # Set with expiration
    cache.set("test_expire", "value", expire_seconds=1)
    assert cache.get("test_expire") == "value"
    
    # Mock Redis TTL
    cache.redis.ttl.return_value = -2  # Key does not exist
    assert cache.get("test_expire") is None

def test_cache_delete(cache: Cache):
    """Test cache deletion."""
    # Set and verify
    cache.set("test_delete", "value")
    assert cache.get("test_delete") == "value"
    
    # Delete and verify
    cache.delete("test_delete")
    assert cache.get("test_delete") is None

def test_cache_clear(cache: Cache):
    """Test cache clearing."""
    # Set multiple keys
    test_data = {
        "key1": "value1",
        "key2": "value2",
        "key3": "value3"
    }
    
    for key, value in test_data.items():
        cache.set(key, value)
    
    # Verify all keys are set
    for key, value in test_data.items():
        assert cache.get(key) == value
    
    # Clear cache
    cache.clear()
    
    # Verify all keys are cleared
    for key in test_data:
        assert cache.get(key) is None

def test_cache_prefix(cache: Cache):
    """Test cache key prefixing."""
    # Set cache with different prefixes
    cache.set("test_key", "value", prefix="prefix1")
    cache.set("test_key", "other_value", prefix="prefix2")
    
    # Verify values with different prefixes
    assert cache.get("test_key", prefix="prefix1") == "value"
    assert cache.get("test_key", prefix="prefix2") == "other_value"

def test_cache_json_serialization(cache: Cache):
    """Test JSON serialization of complex objects."""
    # Complex nested structure
    data = {
        "string": "value",
        "number": 42,
        "float": 3.14,
        "list": [1, 2, 3],
        "dict": {"key": "value"},
        "nested": {
            "list": [{"key": "value"}],
            "dict": {"list": [1, 2, 3]}
        }
    }
    
    # Set data
    cache.set("test_json", data)
    
    # Get and verify data
    result = cache.get("test_json")
    assert result == data
    
    # Verify nested structures
    assert result["nested"]["list"][0]["key"] == "value"
    assert result["nested"]["dict"]["list"] == [1, 2, 3]

def test_cache_none_values(cache: Cache):
    """Test handling of None values."""
    # Set None value
    cache.set("test_none", None)
    
    # Verify None is stored and retrieved correctly
    assert cache.get("test_none") is None
    
    # Verify difference between non-existent key and None value
    assert cache.get("non_existent_key") is None
    assert cache.exists("test_none") is True
    assert cache.exists("non_existent_key") is False

def test_cache_bulk_operations(cache: Cache):
    """Test bulk set and get operations."""
    # Test data
    items = {
        "key1": "value1",
        "key2": 42,
        "key3": {"nested": "value"}
    }
    
    # Bulk set
    cache.bulk_set(items)
    
    # Bulk get
    results = cache.bulk_get(list(items.keys()))
    assert results == items
    
    # Bulk delete
    cache.bulk_delete(list(items.keys()))
    results = cache.bulk_get(list(items.keys()))
    assert all(v is None for v in results.values())

def test_cache_entry():
    """Test CacheEntry class."""
    value = {"key": "value"}
    entry = CacheEntry(value, 1)
    
    assert entry.value == value
    assert not entry.is_expired()
    
    time.sleep(1.1)
    assert entry.is_expired()

def test_local_cache():
    """Test LocalCache class."""
    cache = LocalCache()
    
    # Test set and get
    cache.set("key1", "value1", 1)
    assert cache.get("key1") == "value1"
    
    # Test expiration
    time.sleep(1.1)
    assert cache.get("key1") is None
    
    # Test delete
    cache.set("key2", "value2", 10)
    assert cache.get("key2") == "value2"
    cache.delete("key2")
    assert cache.get("key2") is None
    
    # Test clear
    cache.set("key3", "value3", 10)
    cache.clear()
    assert cache.get("key3") is None

def test_cache_init(mock_redis):
    """Test Cache initialization."""
    cache = Cache(mock_redis)
    assert cache.redis == mock_redis
    
    # Test fallback when Redis fails
    with patch("redis.Redis.from_url", side_effect=RedisError):
        cache = Cache()
        assert cache.redis is None

def test_cache_set_get(mock_redis):
    """Test Cache set and get operations."""
    cache = Cache(mock_redis)
    test_data = {"key": "value"}
    
    # Test successful Redis operation
    mock_redis.get.return_value = json.dumps(test_data).encode()
    assert cache.set("key1", test_data) is True
    assert cache.get("key1") == test_data
    
    # Test Redis failure with local cache fallback
    mock_redis.get.side_effect = RedisError
    assert cache.set("key2", test_data) is True
    assert cache.get("key2") == test_data

def test_cache_delete(mock_redis):
    """Test Cache delete operation."""
    cache = Cache(mock_redis)
    
    # Test successful Redis operation
    assert cache.delete("key1") is True
    mock_redis.delete.assert_called_once_with("key1")
    
    # Test Redis failure with local cache fallback
    mock_redis.delete.side_effect = RedisError
    assert cache.delete("key2") is True

def test_cache_exists(mock_redis):
    """Test Cache exists operation."""
    cache = Cache(mock_redis)
    
    # Test successful Redis operation
    mock_redis.exists.return_value = 1
    assert cache.exists("key1") is True
    mock_redis.exists.assert_called_once_with("key1")
    
    # Test Redis failure with local cache fallback
    mock_redis.exists.side_effect = RedisError
    cache.set("key2", "value2")
    assert cache.exists("key2") is True
    assert not cache.exists("nonexistent")

def test_cache_clear(mock_redis):
    """Test Cache clear operation."""
    cache = Cache(mock_redis)
    
    # Test successful Redis operation
    assert cache.clear() is True
    mock_redis.flushdb.assert_called_once()
    
    # Test Redis failure with local cache fallback
    mock_redis.flushdb.side_effect = RedisError
    assert cache.clear() is True

def test_cache_bulk_operations(mock_redis):
    """Test Cache bulk operations."""
    cache = Cache(mock_redis)
    test_data = {
        "key1": "value1",
        "key2": "value2",
        "key3": "value3"
    }
    
    # Test bulk_set
    mock_redis.pipeline.return_value.__enter__.return_value = MagicMock()
    assert cache.bulk_set(test_data) is True
    
    # Test bulk_get
    mock_redis.get.side_effect = [
        json.dumps(v).encode() for v in test_data.values()
    ]
    result = cache.bulk_get(list(test_data.keys()))
    assert result == test_data
    
    # Test bulk_delete
    assert cache.bulk_delete(list(test_data.keys())) is True
    
    # Test Redis failure with local cache fallback
    mock_redis.pipeline.return_value.__enter__.side_effect = RedisError
    assert cache.bulk_set(test_data) is True
    result = cache.bulk_get(list(test_data.keys()))
    assert result == test_data
    assert cache.bulk_delete(list(test_data.keys())) is True

def test_cache_prefix(mock_redis):
    """Test Cache operations with prefix."""
    cache = Cache(mock_redis)
    test_data = {"key": "value"}
    prefix = "test"
    
    # Test set and get with prefix
    mock_redis.get.return_value = json.dumps(test_data).encode()
    assert cache.set("key1", test_data, prefix=prefix) is True
    assert cache.get("key1", prefix=prefix) == test_data
    mock_redis.get.assert_called_with(f"{prefix}:key1")
    
    # Test delete with prefix
    assert cache.delete("key1", prefix=prefix) is True
    mock_redis.delete.assert_called_with(f"{prefix}:key1")
    
    # Test exists with prefix
    mock_redis.exists.return_value = 1
    assert cache.exists("key1", prefix=prefix) is True
    mock_redis.exists.assert_called_with(f"{prefix}:key1")
