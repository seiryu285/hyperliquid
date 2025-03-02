import pytest
import asyncio
import json
import yaml
import websockets
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from market_data.websocket_client import WebSocketClient
from market_data.stream_manager import StreamManager
from utils.data_validator import DataValidator
from utils.data_normalizer import DataNormalizer
from storage.market_data_store import MarketDataStore

@pytest.fixture
def config():
    return {
        'api': {
            'ws_url': 'wss://api.hyperliquid.xyz/ws',
            'timeout': 30,
            'max_retries': 3,
            'retry_delay': 5
        },
        'websocket': {
            'ping_interval': 30,
            'reconnect_attempts': 5,
            'reconnect_delay': 5,
            'connection_timeout': 60,
            'heartbeat_timeout': 30
        },
        'collection': {
            'symbols': ['BTC-USD', 'ETH-USD'],
            'channels': ['trades', 'orderbook', 'ticker'],
            'orderbook_depth': 20,
            'trade_history_limit': 1000,
            'update_interval': 1
        },
        'processing': {
            'batch_size': 100,
            'max_queue_size': 10000,
            'validation_enabled': True,
            'normalization_enabled': True,
            'missing_data_threshold': 0.1,
            'outlier_std_threshold': 3.0
        },
        'storage': {
            'mongodb': {
                'uri': 'mongodb://localhost:27017',
                'database': 'market_data_test',
                'collections': {
                    'trades': 'trades_test',
                    'orderbook': 'orderbook_test',
                    'ticker': 'ticker_test'
                }
            },
            'compression_enabled': True,
            'retention_days': 30,
            'backup_enabled': True
        }
    }

@pytest.fixture
def sample_trade_data():
    return {
        'type': 'trades',
        'symbol': 'BTC-USD',
        'timestamp': datetime.now().timestamp(),
        'price': 50000.0,
        'size': 1.5,
        'side': 'buy'
    }

@pytest.fixture
def sample_orderbook_data():
    return {
        'type': 'orderbook',
        'symbol': 'BTC-USD',
        'timestamp': datetime.now().timestamp(),
        'bids': [[50000.0, 1.0], [49900.0, 2.0]],
        'asks': [[50100.0, 1.5], [50200.0, 2.5]]
    }

@pytest.fixture
def sample_ticker_data():
    return {
        'type': 'ticker',
        'symbol': 'BTC-USD',
        'timestamp': datetime.now().timestamp(),
        'price': 50000.0,
        'volume': 1000.0,
        'open': 49000.0,
        'high': 51000.0,
        'low': 48000.0,
        'close': 50000.0
    }

class TestWebSocketClient:
    @pytest.mark.asyncio
    async def test_connect(self, config):
        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            ws_client = WebSocketClient(config_path=None)
            ws_client.config = config
            
            mock_ws = AsyncMock()
            mock_connect.return_value = mock_ws
            
            await ws_client.connect()
            
            assert ws_client.connected
            assert ws_client.ws == mock_ws
            mock_connect.assert_called_once_with(
                config['api']['ws_url'],
                ping_interval=config['websocket']['ping_interval'],
                close_timeout=config['websocket']['connection_timeout']
            )

    @pytest.mark.asyncio
    async def test_subscribe(self, config):
        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            ws_client = WebSocketClient(config_path=None)
            ws_client.config = config
            ws_client.ws = AsyncMock()
            ws_client.connected = True
            
            channel = 'trades'
            symbols = ['BTC-USD']
            
            await ws_client.subscribe(channel, symbols)
            
            expected_message = {
                'op': 'subscribe',
                'channel': channel,
                'symbols': symbols
            }
            ws_client.ws.send.assert_called_once_with(json.dumps(expected_message))
            assert frozenset({'channel': channel, 'symbols': tuple(symbols)}.items()) in ws_client.subscriptions

class TestStreamManager:
    @pytest.mark.asyncio
    async def test_start_stop(self, config):
        with patch('market_data.websocket_client.WebSocketClient') as mock_ws_client:
            stream_manager = StreamManager(config_path=None)
            stream_manager.ws_client = mock_ws_client
            
            await stream_manager.start()
            assert stream_manager.running
            assert len(stream_manager.tasks) == 4
            
            await stream_manager.stop()
            assert not stream_manager.running
            for task in stream_manager.tasks:
                assert task.cancelled()

class TestDataValidator:
    @pytest.mark.asyncio
    async def test_validate_trade(self, config, sample_trade_data):
        validator = DataValidator(config_path=None)
        validator.config = config
        
        assert await validator.validate_message(sample_trade_data)
        
        # Test invalid price
        invalid_data = sample_trade_data.copy()
        invalid_data['price'] = -1
        assert not await validator.validate_message(invalid_data)
        
        # Test invalid size
        invalid_data = sample_trade_data.copy()
        invalid_data['size'] = 0
        assert not await validator.validate_message(invalid_data)
        
        # Test invalid side
        invalid_data = sample_trade_data.copy()
        invalid_data['side'] = 'invalid'
        assert not await validator.validate_message(invalid_data)

    @pytest.mark.asyncio
    async def test_validate_orderbook(self, config, sample_orderbook_data):
        validator = DataValidator(config_path=None)
        validator.config = config
        
        assert await validator.validate_message(sample_orderbook_data)
        
        # Test invalid price levels
        invalid_data = sample_orderbook_data.copy()
        invalid_data['bids'] = [[-1, 1.0]]
        assert not await validator.validate_message(invalid_data)
        
        # Test price ordering
        invalid_data = sample_orderbook_data.copy()
        invalid_data['bids'] = [[50300.0, 1.0]]  # Higher than asks
        assert not await validator.validate_message(invalid_data)

class TestDataNormalizer:
    @pytest.mark.asyncio
    async def test_normalize_trade(self, config, sample_trade_data):
        normalizer = DataNormalizer(config_path=None)
        normalizer.config = config
        
        # Add some history
        symbol = sample_trade_data['symbol']
        normalizer.price_history[symbol] = [50000.0] * 10
        normalizer.volume_history[symbol] = [1.5] * 10
        
        normalized_data = await normalizer.normalize_data(sample_trade_data)
        
        assert 'normalized_price' in normalized_data
        assert 'normalized_size' in normalized_data
        assert isinstance(normalized_data['normalized_price'], float)
        assert isinstance(normalized_data['normalized_size'], float)

    @pytest.mark.asyncio
    async def test_normalize_orderbook(self, config, sample_orderbook_data):
        normalizer = DataNormalizer(config_path=None)
        normalizer.config = config
        
        normalized_data = await normalizer.normalize_data(sample_orderbook_data)
        
        assert 'normalized_bids' in normalized_data
        assert 'normalized_asks' in normalized_data
        assert len(normalized_data['normalized_bids']) == len(sample_orderbook_data['bids'])
        assert len(normalized_data['normalized_asks']) == len(sample_orderbook_data['asks'])

class TestMarketDataStore:
    @pytest.mark.asyncio
    async def test_store_and_retrieve_data(self, config, sample_trade_data):
        store = MarketDataStore(config_path=None)
        store.config = config
        
        # Store data
        await store.store_data(sample_trade_data)
        await store.flush()
        
        # Retrieve latest data
        latest_data = await store.get_latest_data(
            sample_trade_data['symbol'],
            sample_trade_data['type']
        )
        
        assert latest_data is not None
        assert latest_data['symbol'] == sample_trade_data['symbol']
        assert latest_data['price'] == sample_trade_data['price']

    @pytest.mark.asyncio
    async def test_historical_data(self, config, sample_trade_data):
        store = MarketDataStore(config_path=None)
        store.config = config
        
        # Store multiple data points
        for i in range(5):
            data = sample_trade_data.copy()
            data['timestamp'] = (datetime.now() - timedelta(minutes=i)).timestamp()
            await store.store_data(data)
        await store.flush()
        
        # Retrieve historical data
        start_time = datetime.now() - timedelta(minutes=5)
        end_time = datetime.now()
        historical_data = await store.get_historical_data(
            sample_trade_data['symbol'],
            sample_trade_data['type'],
            start_time,
            end_time
        )
        
        assert len(historical_data) == 5
        assert all(d['symbol'] == sample_trade_data['symbol'] for d in historical_data)

    @pytest.mark.asyncio
    async def test_batch_processing(self, config, sample_trade_data):
        store = MarketDataStore(config_path=None)
        store.config = config
        store.config['processing']['batch_size'] = 3
        
        # Store multiple data points
        for i in range(5):
            data = sample_trade_data.copy()
            data['timestamp'] = (datetime.now() - timedelta(minutes=i)).timestamp()
            await store.store_data(data)
        
        # First batch should be automatically flushed
        collection_name = config['storage']['mongodb']['collections']['trades']
        assert len(store.batch_data.get(collection_name, [])) == 2  # 5 % 3 = 2 remaining
        
        # Flush remaining data
        await store.flush()
        assert len(store.batch_data.get(collection_name, [])) == 0
