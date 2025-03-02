import asyncio
import logging
import pytest
from market_data.websocket_client import WebSocketClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_websocket_connection():
    """WebSocket接続のテスト"""
    try:
        client = WebSocketClient()
        await client.connect()
        
        # 接続状態の確認
        assert client.connected, "WebSocket connection failed"
        logger.info("WebSocket connection successful")
        
        # 市場データの購読
        await client.subscribe("market_data", ["BTC-PERP"])
        logger.info("Subscribed to BTC-PERP")
        
        # メッセージの受信テスト（30秒間）
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < 30:
            message = await client.message_queue.get()
            logger.info(f"Received message: {message}")
            
            # データの検証
            is_valid, error = client.validate_market_data(message)
            assert is_valid, f"Invalid market data: {error}"
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        if hasattr(client, 'ws') and client.ws:
            await client.ws.close()
        logger.info("Test completed")
