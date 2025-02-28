"""
HyperLiquid Market Data Collector

This module is responsible for collecting real-time market data from HyperLiquid testnet
via WebSocket and REST API. It stores the data in MongoDB for further processing.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

import aiohttp
import motor.motor_asyncio
import websockets
from pymongo import ASCENDING, DESCENDING

from core.config import settings
from core.auth import HyperLiquidAuth

# Configure logging
logger = logging.getLogger(__name__)

class HyperLiquidDataCollector:
    """
    Data collector for HyperLiquid market data.
    Collects and stores price, orderbook, and candle data.
    """
    
    def __init__(self, db_client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None):
        """
        Initialize the data collector.
        
        Args:
            db_client: MongoDB client (optional, will create one if not provided)
        """
        self.auth = HyperLiquidAuth()
        self.rest_base_url = settings.HYPERLIQUID_API_URL
        self.ws_url = settings.HYPERLIQUID_WS_URL
        
        # Initialize MongoDB client
        if db_client is None:
            self.client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGODB_URL)
        else:
            self.client = db_client
            
        self.db = self.client[settings.MONGO_DB]
        
        # Collections
        self.trades_collection = self.db.trades
        self.orderbook_collection = self.db.orderbook
        self.candles_collection = self.db.candles
        self.ticker_collection = self.db.ticker
        
        # Create indexes
        asyncio.create_task(self._create_indexes())
        
        # Active subscriptions
        self.active_subscriptions = set()
        self.ws_connection = None
        self.is_running = False
        
    async def _create_indexes(self):
        """Create MongoDB indexes for efficient queries."""
        await self.trades_collection.create_index([("symbol", ASCENDING), ("timestamp", DESCENDING)])
        await self.orderbook_collection.create_index([("symbol", ASCENDING), ("timestamp", DESCENDING)])
        await self.candles_collection.create_index([("symbol", ASCENDING), ("interval", ASCENDING), ("timestamp", DESCENDING)])
        await self.ticker_collection.create_index([("symbol", ASCENDING), ("timestamp", DESCENDING)])
        logger.info("Created MongoDB indexes for market data collections")
    
    async def _make_rest_request(self, endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Dict:
        """
        Make a REST API request to HyperLiquid.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            data: Request data
            
        Returns:
            API response as dictionary
        """
        url = f"{self.rest_base_url}{endpoint}"
        headers = self.auth.get_auth_headers(method, endpoint, data)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, headers=headers, json=data) as response:
                    if response.status >= 400:
                        text = await response.text()
                        logger.error(f"API error: {response.status} - {text}")
                        return {"error": f"API error: {response.status}", "message": text}
                    
                    return await response.json()
        except Exception as e:
            logger.error(f"REST request error: {e}")
            return {"error": "Request failed", "message": str(e)}
    
    async def connect_websocket(self):
        """Establish WebSocket connection to HyperLiquid."""
        try:
            self.ws_connection = await websockets.connect(self.ws_url)
            logger.info("Connected to HyperLiquid WebSocket")
            return True
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            return False
    
    async def subscribe_trades(self, symbols: List[str]):
        """
        Subscribe to trade updates for specified symbols.
        
        Args:
            symbols: List of trading pairs to subscribe to
        """
        if not self.ws_connection:
            if not await self.connect_websocket():
                return
        
        for symbol in symbols:
            subscription = {
                "method": "subscribe",
                "params": {
                    "channel": "trades",
                    "symbol": symbol
                }
            }
            
            await self.ws_connection.send(json.dumps(subscription))
            self.active_subscriptions.add(f"trades:{symbol}")
            logger.info(f"Subscribed to trades for {symbol}")
    
    async def subscribe_orderbook(self, symbols: List[str], depth: int = 20):
        """
        Subscribe to orderbook updates for specified symbols.
        
        Args:
            symbols: List of trading pairs to subscribe to
            depth: Orderbook depth
        """
        if not self.ws_connection:
            if not await self.connect_websocket():
                return
        
        for symbol in symbols:
            subscription = {
                "method": "subscribe",
                "params": {
                    "channel": "orderbook",
                    "symbol": symbol,
                    "depth": depth
                }
            }
            
            await self.ws_connection.send(json.dumps(subscription))
            self.active_subscriptions.add(f"orderbook:{symbol}")
            logger.info(f"Subscribed to orderbook for {symbol} with depth {depth}")
    
    async def subscribe_candles(self, symbols: List[str], interval: str = "1m"):
        """
        Subscribe to candle updates for specified symbols.
        
        Args:
            symbols: List of trading pairs to subscribe to
            interval: Candle interval (e.g., "1m", "5m", "1h", "1d")
        """
        if not self.ws_connection:
            if not await self.connect_websocket():
                return
        
        for symbol in symbols:
            subscription = {
                "method": "subscribe",
                "params": {
                    "channel": "candles",
                    "symbol": symbol,
                    "interval": interval
                }
            }
            
            await self.ws_connection.send(json.dumps(subscription))
            self.active_subscriptions.add(f"candles:{symbol}:{interval}")
            logger.info(f"Subscribed to {interval} candles for {symbol}")
    
    async def _process_trade_message(self, message: Dict):
        """
        Process and store trade message.
        
        Args:
            message: Trade message from WebSocket
        """
        try:
            symbol = message.get("symbol")
            trades = message.get("data", [])
            
            if not trades:
                return
            
            # Format trades for MongoDB
            formatted_trades = []
            for trade in trades:
                formatted_trade = {
                    "symbol": symbol,
                    "price": float(trade.get("price")),
                    "size": float(trade.get("size")),
                    "side": trade.get("side"),
                    "timestamp": datetime.fromtimestamp(trade.get("timestamp") / 1000),
                    "trade_id": trade.get("id"),
                    "raw_data": trade
                }
                formatted_trades.append(formatted_trade)
            
            # Insert trades into MongoDB
            if formatted_trades:
                await self.trades_collection.insert_many(formatted_trades)
                logger.debug(f"Stored {len(formatted_trades)} trades for {symbol}")
        except Exception as e:
            logger.error(f"Error processing trade message: {e}")
    
    async def _process_orderbook_message(self, message: Dict):
        """
        Process and store orderbook message.
        
        Args:
            message: Orderbook message from WebSocket
        """
        try:
            symbol = message.get("symbol")
            data = message.get("data", {})
            
            if not data:
                return
            
            # Format orderbook for MongoDB
            formatted_orderbook = {
                "symbol": symbol,
                "timestamp": datetime.utcnow(),
                "bids": data.get("bids", []),
                "asks": data.get("asks", []),
                "raw_data": data
            }
            
            # Insert orderbook into MongoDB
            await self.orderbook_collection.insert_one(formatted_orderbook)
            logger.debug(f"Stored orderbook for {symbol}")
        except Exception as e:
            logger.error(f"Error processing orderbook message: {e}")
    
    async def _process_candle_message(self, message: Dict):
        """
        Process and store candle message.
        
        Args:
            message: Candle message from WebSocket
        """
        try:
            symbol = message.get("symbol")
            interval = message.get("interval")
            candles = message.get("data", [])
            
            if not candles:
                return
            
            # Format candles for MongoDB
            formatted_candles = []
            for candle in candles:
                formatted_candle = {
                    "symbol": symbol,
                    "interval": interval,
                    "timestamp": datetime.fromtimestamp(candle[0] / 1000),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5]),
                    "raw_data": candle
                }
                formatted_candles.append(formatted_candle)
            
            # Insert candles into MongoDB
            if formatted_candles:
                await self.candles_collection.insert_many(formatted_candles)
                logger.debug(f"Stored {len(formatted_candles)} candles for {symbol} ({interval})")
        except Exception as e:
            logger.error(f"Error processing candle message: {e}")
    
    async def _handle_websocket_messages(self):
        """Handle incoming WebSocket messages."""
        if not self.ws_connection:
            logger.error("No WebSocket connection available")
            return
        
        try:
            while self.is_running:
                message = await self.ws_connection.recv()
                message = json.loads(message)
                
                channel = message.get("channel")
                
                if channel == "trades":
                    await self._process_trade_message(message)
                elif channel == "orderbook":
                    await self._process_orderbook_message(message)
                elif channel == "candles":
                    await self._process_candle_message(message)
                else:
                    logger.warning(f"Received message from unknown channel: {channel}")
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def fetch_historical_candles(self, symbol: str, interval: str = "1h", limit: int = 1000, 
                                      start_time: Optional[int] = None, end_time: Optional[int] = None):
        """
        Fetch historical candles from REST API.
        
        Args:
            symbol: Trading pair
            interval: Candle interval
            limit: Number of candles to fetch
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            
        Returns:
            List of candles
        """
        endpoint = f"/api/candles"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        if start_time:
            params["startTime"] = start_time
        
        if end_time:
            params["endTime"] = end_time
        
        response = await self._make_rest_request(endpoint, data=params)
        
        if "error" in response:
            logger.error(f"Error fetching historical candles: {response}")
            return []
        
        # Format and store candles
        formatted_candles = []
        for candle in response:
            formatted_candle = {
                "symbol": symbol,
                "interval": interval,
                "timestamp": datetime.fromtimestamp(candle[0] / 1000),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5]),
                "raw_data": candle
            }
            formatted_candles.append(formatted_candle)
        
        # Store in MongoDB
        if formatted_candles:
            await self.candles_collection.insert_many(formatted_candles)
            logger.info(f"Stored {len(formatted_candles)} historical candles for {symbol} ({interval})")
        
        return formatted_candles
    
    async def fetch_ticker(self, symbols: List[str]):
        """
        Fetch current ticker data for specified symbols.
        
        Args:
            symbols: List of trading pairs
            
        Returns:
            Dictionary mapping symbols to ticker data
        """
        endpoint = "/api/ticker"
        
        response = await self._make_rest_request(endpoint)
        
        if "error" in response:
            logger.error(f"Error fetching ticker data: {response}")
            return {}
        
        # Filter and store ticker data for requested symbols
        ticker_data = {}
        timestamp = datetime.utcnow()
        
        for symbol_data in response:
            symbol = symbol_data.get("symbol")
            
            if symbol in symbols:
                ticker_data[symbol] = symbol_data
                
                # Store in MongoDB
                formatted_ticker = {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "last_price": float(symbol_data.get("lastPrice", 0)),
                    "price_change": float(symbol_data.get("priceChange", 0)),
                    "price_change_percent": float(symbol_data.get("priceChangePercent", 0)),
                    "volume_24h": float(symbol_data.get("volume", 0)),
                    "high_24h": float(symbol_data.get("highPrice", 0)),
                    "low_24h": float(symbol_data.get("lowPrice", 0)),
                    "raw_data": symbol_data
                }
                
                await self.ticker_collection.insert_one(formatted_ticker)
        
        logger.info(f"Fetched and stored ticker data for {len(ticker_data)} symbols")
        return ticker_data
    
    async def start(self, symbols: List[str], candle_intervals: List[str] = ["1m", "5m", "1h", "4h", "1d"]):
        """
        Start the data collector.
        
        Args:
            symbols: List of trading pairs to collect data for
            candle_intervals: List of candle intervals to collect
        """
        if self.is_running:
            logger.warning("Data collector is already running")
            return
        
        self.is_running = True
        
        # Connect to WebSocket
        if not await self.connect_websocket():
            self.is_running = False
            return
        
        # Subscribe to channels
        await self.subscribe_trades(symbols)
        await self.subscribe_orderbook(symbols)
        
        for interval in candle_intervals:
            await self.subscribe_candles(symbols, interval)
        
        # Start message handler
        asyncio.create_task(self._handle_websocket_messages())
        
        # Start periodic ticker fetching
        asyncio.create_task(self._periodic_ticker_fetch(symbols))
        
        logger.info(f"Data collector started for symbols: {symbols}")
    
    async def _periodic_ticker_fetch(self, symbols: List[str], interval_seconds: int = 60):
        """
        Periodically fetch ticker data.
        
        Args:
            symbols: List of trading pairs
            interval_seconds: Fetch interval in seconds
        """
        while self.is_running:
            try:
                await self.fetch_ticker(symbols)
            except Exception as e:
                logger.error(f"Error in periodic ticker fetch: {e}")
            
            await asyncio.sleep(interval_seconds)
    
    async def stop(self):
        """Stop the data collector."""
        self.is_running = False
        
        if self.ws_connection:
            await self.ws_connection.close()
            self.ws_connection = None
        
        logger.info("Data collector stopped")

# Singleton instance
data_collector = None

async def get_data_collector() -> HyperLiquidDataCollector:
    """
    Get or create the data collector singleton instance.
    
    Returns:
        HyperLiquidDataCollector instance
    """
    global data_collector
    
    if data_collector is None:
        client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGODB_URL)
        data_collector = HyperLiquidDataCollector(client)
    
    return data_collector

if __name__ == "__main__":
    # Example usage
    async def main():
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Get data collector
        collector = await get_data_collector()
        
        # Start collecting data for BTC and ETH
        symbols = ["BTC-USD", "ETH-USD"]
        await collector.start(symbols)
        
        # Run for 1 hour
        await asyncio.sleep(3600)
        
        # Stop collector
        await collector.stop()
    
    # Run the example
    asyncio.run(main())
