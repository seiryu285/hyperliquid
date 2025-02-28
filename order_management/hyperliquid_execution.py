"""
Hyperliquid-specific execution engine.

This module provides an execution engine implementation for the
Hyperliquid exchange.
"""

import logging
import asyncio
import json
import time
import hmac
import hashlib
import base64
from typing import Dict, List, Optional, Any, Callable, Coroutine, Union, Tuple
from datetime import datetime

import aiohttp
from pydantic import BaseModel

from .order_types import (
    Order, OrderRequest, OrderCancelRequest, OrderFill, 
    OrderStatus, OrderSide, OrderType, Position
)
from .order_manager import OrderManager
from .execution_engine import ExecutionEngine
from core.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()


class HyperliquidExecutionEngine(ExecutionEngine):
    """
    Execution engine for the Hyperliquid exchange.
    
    This class implements the execution engine interface for the
    Hyperliquid exchange.
    """
    
    def __init__(self, 
                order_manager: OrderManager,
                api_key: Optional[str] = None,
                api_secret: Optional[str] = None,
                base_url: Optional[str] = None,
                testnet: bool = True,
                max_concurrent_requests: int = 10,
                request_timeout: float = 10.0,
                rate_limit_per_second: float = 5.0):
        """
        Initialize the Hyperliquid execution engine.
        
        Args:
            order_manager: Order manager instance
            api_key: API key for authentication
            api_secret: API secret for authentication
            base_url: Base URL for the API
            testnet: Whether to use the testnet
            max_concurrent_requests: Maximum number of concurrent requests
            request_timeout: Request timeout in seconds
            rate_limit_per_second: Maximum number of requests per second
        """
        super().__init__(order_manager)
        
        # API credentials
        self.api_key = api_key or settings.HYPERLIQUID_API_KEY
        self.api_secret = api_secret or settings.HYPERLIQUID_API_SECRET
        
        # API URLs
        if base_url:
            self.base_url = base_url
        elif testnet:
            self.base_url = "https://api.hyperliquid-testnet.xyz"
        else:
            self.base_url = "https://api.hyperliquid.xyz"
        
        # REST API endpoints
        self.rest_url = f"{self.base_url}/rest"
        
        # HTTP session
        self.session = None
        
        # Order mapping (client order ID -> exchange order ID)
        self.order_id_mapping = {}
        
        # Concurrency and rate limiting settings
        self.max_concurrent_requests = max_concurrent_requests
        self.request_timeout = request_timeout
        self.rate_limit_per_second = rate_limit_per_second
        self.request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.last_request_time = 0
        self.request_interval = 1.0 / rate_limit_per_second if rate_limit_per_second > 0 else 0
        
        # Error tracking
        self.error_counts = {}
        self.max_error_threshold = 10  # Maximum number of errors before temporary blacklisting
        self.error_cooldown = 300  # Cooldown period in seconds after reaching error threshold
        self.error_blacklist = {}  # Endpoints temporarily blacklisted due to errors
        
        logger.info(f"Initialized Hyperliquid execution engine with base URL: {self.base_url}, " 
                   f"max_concurrent_requests: {max_concurrent_requests}, "
                   f"rate_limit: {rate_limit_per_second}/s")
    
    async def start(self):
        """Start the execution engine."""
        await super().start()
        
        # Create HTTP session
        self.session = aiohttp.ClientSession()
        
        logger.info("Started Hyperliquid execution engine")
    
    async def stop(self):
        """Stop the execution engine."""
        await super().stop()
        
        # Close HTTP session
        if self.session:
            await self.session.close()
            self.session = None
        
        logger.info("Stopped Hyperliquid execution engine")
    
    async def submit_order(self, order: Order) -> Tuple[bool, Optional[str]]:
        """
        Submit an order to the exchange.
        
        Args:
            order: Order to submit
            
        Returns:
            Tuple of (success, error_message)
        """
        # Add order to order manager
        await self.order_manager.add_order_async(order)
        
        # Submit order to exchange
        try:
            success, error = await self.order_manager.retry_async(
                self._submit_order_to_exchange, order
            )
            
            if not success:
                # Update order status to failed
                await self.order_manager.update_order_async(order.id, {
                    'status': OrderStatus.FAILED,
                    'error_message': error
                })
                return False, error
            
            # Update order status to submitted
            await self.order_manager.update_order_async(order.id, {
                'status': OrderStatus.SUBMITTED
            })
            
            return True, None
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            
            # Update order status to failed
            await self.order_manager.update_order_async(order.id, {
                'status': OrderStatus.FAILED,
                'error_message': str(e)
            })
            
            return False, str(e)

    async def cancel_order(self, order_id: str) -> Tuple[bool, Optional[str]]:
        """
        Cancel an order on the exchange.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Tuple of (success, error_message)
        """
        order = self.order_manager.get_order(order_id)
        if not order:
            return False, f"Order {order_id} not found"
        
        try:
            success, error = await self.order_manager.retry_async(
                self._cancel_order_on_exchange, order
            )
            
            if not success:
                return False, error
            
            # Update order status to cancelled
            await self.order_manager.cancel_order_async(order_id)
            
            return True, None
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False, str(e)

    async def _submit_order_to_exchange(self, order: Order) -> Tuple[bool, Optional[str]]:
        """
        Submit an order to the Hyperliquid exchange.
        
        Args:
            order: Order to submit
            
        Returns:
            Tuple of (success, error_message)
        """
        if not self.session:
            return False, "Execution engine not started"
        
        try:
            # Convert order to Hyperliquid format
            hl_order = self._convert_order_to_hyperliquid(order)
            
            # Sign the request
            signature = self._sign_request(hl_order)
            
            # Prepare headers
            headers = {
                'Content-Type': 'application/json',
                'X-API-Key': self.api_key,
                'X-API-Signature': signature
            }
            
            # Rate limit and concurrency control
            await self._rate_limit()
            
            # Send the request with timeout and concurrency control
            async with self.request_semaphore:
                async with self.session.post(
                    f"{self.rest_url}/exchange/order",
                    json=hl_order,
                    headers=headers,
                    timeout=self.request_timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Error submitting order: HTTP {response.status}: {error_text}")
                        self._track_error(f"{self.rest_url}/exchange/order")
                        return False, f"HTTP error {response.status}: {error_text}"
                    
                    # Parse response
                    response_data = await response.json()
                    
                    # Check for errors
                    if 'error' in response_data:
                        logger.error(f"Error submitting order: {response_data['error']}")
                        self._track_error(f"{self.rest_url}/exchange/order")
                        return False, response_data['error']
                    
                    # Extract exchange order ID
                    exchange_id = response_data.get('orderId')
                    if not exchange_id:
                        logger.error("No order ID in response")
                        return False, "No order ID in response"
                    
                    # Update order with exchange ID
                    await self.order_manager.update_order_async(order.id, {
                        'exchange_id': exchange_id
                    })
                    
                    # Store mapping
                    self.order_id_mapping[order.id] = exchange_id
                    
                    logger.info(f"Order {order.id} submitted to Hyperliquid with exchange ID {exchange_id}")
                    
                    # Reset error count for successful request
                    self._reset_error(f"{self.rest_url}/exchange/order")
                    
                    return True, None
        
        except asyncio.TimeoutError:
            logger.error(f"Timeout submitting order after {self.request_timeout}s")
            self._track_error(f"{self.rest_url}/exchange/order")
            return False, f"Request timeout after {self.request_timeout}s"
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            self._track_error(f"{self.rest_url}/exchange/order")
            return False, str(e)

    async def _cancel_order_on_exchange(self, order: Order) -> Tuple[bool, Optional[str]]:
        """
        Cancel an order on the Hyperliquid exchange.
        
        Args:
            order: Order to cancel
            
        Returns:
            Tuple of (success, error_message)
        """
        if not self.session:
            return False, "Execution engine not started"
        
        try:
            # Get exchange order ID
            exchange_id = order.exchange_id or self.order_id_mapping.get(order.id)
            if not exchange_id:
                return False, "No exchange order ID"
            
            # Prepare cancel request
            cancel_request = {
                'orderId': exchange_id,
                'symbol': order.symbol
            }
            
            # Sign the request
            signature = self._sign_request(cancel_request)
            
            # Prepare headers
            headers = {
                'Content-Type': 'application/json',
                'X-API-Key': self.api_key,
                'X-API-Signature': signature
            }
            
            # Rate limit and concurrency control
            await self._rate_limit()
            
            # Send the request with timeout and concurrency control
            async with self.request_semaphore:
                async with self.session.post(
                    f"{self.rest_url}/exchange/cancel",
                    json=cancel_request,
                    headers=headers,
                    timeout=self.request_timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Error cancelling order: HTTP {response.status}: {error_text}")
                        self._track_error(f"{self.rest_url}/exchange/cancel")
                        return False, f"HTTP error {response.status}: {error_text}"
                    
                    # Parse response
                    response_data = await response.json()
                    
                    # Check for errors
                    if 'error' in response_data:
                        logger.error(f"Error cancelling order: {response_data['error']}")
                        self._track_error(f"{self.rest_url}/exchange/cancel")
                        return False, response_data['error']
                    
                    logger.info(f"Order {order.id} cancelled on Hyperliquid")
                    
                    # Reset error count for successful request
                    self._reset_error(f"{self.rest_url}/exchange/cancel")
                    
                    return True, None
        
        except asyncio.TimeoutError:
            logger.error(f"Timeout cancelling order after {self.request_timeout}s")
            self._track_error(f"{self.rest_url}/exchange/cancel")
            return False, f"Request timeout after {self.request_timeout}s"
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            self._track_error(f"{self.rest_url}/exchange/cancel")
            return False, str(e)

    async def _rate_limit(self):
        """Apply rate limiting to API requests."""
        if self.request_interval > 0:
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            
            if time_since_last_request < self.request_interval:
                # Wait until we can make the next request
                await asyncio.sleep(self.request_interval - time_since_last_request)
            
            self.last_request_time = time.time()

    def _track_error(self, endpoint: str):
        """
        Track errors for an endpoint.
        
        Args:
            endpoint: API endpoint
        """
        # Initialize error count if not present
        if endpoint not in self.error_counts:
            self.error_counts[endpoint] = 0
        
        # Increment error count
        self.error_counts[endpoint] += 1
        
        # Check if we've reached the threshold
        if self.error_counts[endpoint] >= self.max_error_threshold:
            logger.warning(f"Endpoint {endpoint} has reached error threshold, blacklisting for {self.error_cooldown}s")
            self.error_blacklist[endpoint] = time.time() + self.error_cooldown
            self.error_counts[endpoint] = 0

    def _reset_error(self, endpoint: str):
        """
        Reset error count for an endpoint.
        
        Args:
            endpoint: API endpoint
        """
        self.error_counts[endpoint] = 0
        
        # Remove from blacklist if present
        if endpoint in self.error_blacklist:
            del self.error_blacklist[endpoint]

    def _is_blacklisted(self, endpoint: str) -> bool:
        """
        Check if an endpoint is blacklisted.
        
        Args:
            endpoint: API endpoint
            
        Returns:
            True if blacklisted, False otherwise
        """
        if endpoint in self.error_blacklist:
            # Check if cooldown period has expired
            if time.time() > self.error_blacklist[endpoint]:
                # Remove from blacklist
                del self.error_blacklist[endpoint]
                return False
            
            return True
        
        return False

    async def _get_order_status_from_exchange(self, order: Order) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """
        Get the status of an order from the Hyperliquid exchange.
        
        Args:
            order: Order to get status for
            
        Returns:
            Tuple of (success, status_updates, error_message)
        """
        if not self.session:
            return False, {}, "Execution engine not started"
        
        try:
            # Get exchange order ID
            exchange_id = order.exchange_id or self.order_id_mapping.get(order.id)
            if not exchange_id:
                return False, {}, "No exchange order ID"
            
            # Prepare request
            request_data = {
                'orderId': exchange_id,
                'symbol': order.symbol
            }
            
            # Sign the request
            signature = self._sign_request(request_data)
            
            # Prepare headers
            headers = {
                'Content-Type': 'application/json',
                'X-API-Key': self.api_key,
                'X-API-Signature': signature
            }
            
            # Rate limit and concurrency control
            await self._rate_limit()
            
            # Send the request with timeout and concurrency control
            async with self.request_semaphore:
                async with self.session.post(
                    f"{self.rest_url}/exchange/order/status",
                    json=request_data,
                    headers=headers,
                    timeout=self.request_timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Error getting order status: HTTP {response.status}: {error_text}")
                        self._track_error(f"{self.rest_url}/exchange/order/status")
                        return False, {}, f"HTTP error {response.status}: {error_text}"
                    
                    # Parse response
                    response_data = await response.json()
                    
                    # Check for errors
                    if 'error' in response_data:
                        logger.error(f"Error getting order status: {response_data['error']}")
                        self._track_error(f"{self.rest_url}/exchange/order/status")
                        return False, {}, response_data['error']
                    
                    # Extract status information
                    status_updates = self._parse_order_status(response_data, order)
                    
                    logger.info(f"Updated status for order {order.id}")
                    
                    # Reset error count for successful request
                    self._reset_error(f"{self.rest_url}/exchange/order/status")
                    
                    return True, status_updates, None
        
        except asyncio.TimeoutError:
            logger.error(f"Timeout getting order status after {self.request_timeout}s")
            self._track_error(f"{self.rest_url}/exchange/order/status")
            return False, {}, f"Request timeout after {self.request_timeout}s"
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            self._track_error(f"{self.rest_url}/exchange/order/status")
            return False, {}, str(e)

    async def get_open_orders(self, symbol: Optional[str] = None) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
        """
        Get open orders from the exchange.
        
        Args:
            symbol: Symbol to filter by
            
        Returns:
            Tuple of (success, orders, error_message)
        """
        if not self.session:
            return False, [], "Execution engine not started"
        
        try:
            # Prepare request
            request_data = {}
            if symbol:
                request_data['symbol'] = symbol
            
            # Sign the request
            signature = self._sign_request(request_data)
            
            # Prepare headers
            headers = {
                'Content-Type': 'application/json',
                'X-API-Key': self.api_key,
                'X-API-Signature': signature
            }
            
            # Rate limit and concurrency control
            await self._rate_limit()
            
            # Send the request with timeout and concurrency control
            async with self.request_semaphore:
                async with self.session.post(
                    f"{self.rest_url}/exchange/orders",
                    json=request_data,
                    headers=headers,
                    timeout=self.request_timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Error getting open orders: HTTP {response.status}: {error_text}")
                        self._track_error(f"{self.rest_url}/exchange/orders")
                        return False, [], f"HTTP error {response.status}: {error_text}"
                    
                    # Parse response
                    response_data = await response.json()
                    
                    # Check for errors
                    if 'error' in response_data:
                        logger.error(f"Error getting open orders: {response_data['error']}")
                        self._track_error(f"{self.rest_url}/exchange/orders")
                        return False, [], response_data['error']
                    
                    # Extract orders
                    orders = response_data.get('orders', [])
                    
                    logger.info(f"Retrieved {len(orders)} open orders from Hyperliquid")
                    
                    # Reset error count for successful request
                    self._reset_error(f"{self.rest_url}/exchange/orders")
                    
                    return True, orders, None
        
        except asyncio.TimeoutError:
            logger.error(f"Timeout getting open orders after {self.request_timeout}s")
            self._track_error(f"{self.rest_url}/exchange/orders")
            return False, [], f"Request timeout after {self.request_timeout}s"
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            self._track_error(f"{self.rest_url}/exchange/orders")
            return False, [], str(e)

    async def get_positions(self) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
        """
        Get positions from the exchange.
        
        Returns:
            Tuple of (success, positions, error_message)
        """
        if not self.session:
            return False, [], "Execution engine not started"
        
        try:
            # Prepare request
            request_data = {}
            
            # Sign the request
            signature = self._sign_request(request_data)
            
            # Prepare headers
            headers = {
                'Content-Type': 'application/json',
                'X-API-Key': self.api_key,
                'X-API-Signature': signature
            }
            
            # Rate limit and concurrency control
            await self._rate_limit()
            
            # Send the request with timeout and concurrency control
            async with self.request_semaphore:
                async with self.session.post(
                    f"{self.rest_url}/exchange/positions",
                    json=request_data,
                    headers=headers,
                    timeout=self.request_timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Error getting positions: HTTP {response.status}: {error_text}")
                        self._track_error(f"{self.rest_url}/exchange/positions")
                        return False, [], f"HTTP error {response.status}: {error_text}"
                    
                    # Parse response
                    response_data = await response.json()
                    
                    # Check for errors
                    if 'error' in response_data:
                        logger.error(f"Error getting positions: {response_data['error']}")
                        self._track_error(f"{self.rest_url}/exchange/positions")
                        return False, [], response_data['error']
                    
                    # Extract positions
                    positions = response_data.get('positions', [])
                    
                    logger.info(f"Retrieved {len(positions)} positions from Hyperliquid")
                    
                    # Reset error count for successful request
                    self._reset_error(f"{self.rest_url}/exchange/positions")
                    
                    return True, positions, None
        
        except asyncio.TimeoutError:
            logger.error(f"Timeout getting positions after {self.request_timeout}s")
            self._track_error(f"{self.rest_url}/exchange/positions")
            return False, [], f"Request timeout after {self.request_timeout}s"
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            self._track_error(f"{self.rest_url}/exchange/positions")
            return False, [], str(e)

    async def update_positions(self) -> bool:
        """
        Update positions from the exchange.
        
        Returns:
            True if successful, False otherwise
        """
        success, positions, error = await self.get_positions()
        if not success:
            logger.error(f"Failed to update positions: {error}")
            return False
        
        # Update positions in order manager
        for pos_data in positions:
            symbol = pos_data.get('symbol')
            if not symbol:
                continue
            
            # Extract position data
            quantity = float(pos_data.get('size', 0))
            entry_price = float(pos_data.get('entryPrice', 0))
            current_price = float(pos_data.get('markPrice', 0))
            unrealized_pnl = float(pos_data.get('unrealizedPnl', 0))
            realized_pnl = float(pos_data.get('realizedPnl', 0))
            
            # Update position
            self.order_manager.update_position(symbol, {
                'quantity': quantity,
                'entry_price': entry_price,
                'current_price': current_price,
                'unrealized_pnl': unrealized_pnl,
                'realized_pnl': realized_pnl
            })
        
        return True
    
    def _convert_order_to_hyperliquid(self, order: Order) -> Dict[str, Any]:
        """
        Convert an order to Hyperliquid format.
        
        Args:
            order: Order to convert
            
        Returns:
            Order in Hyperliquid format
        """
        # Convert order side
        side = "buy" if order.side == OrderSide.BUY else "sell"
        
        # Convert order type
        if order.type == OrderType.MARKET:
            type_str = "market"
        elif order.type == OrderType.LIMIT:
            type_str = "limit"
        elif order.type == OrderType.STOP:
            type_str = "stop"
        elif order.type == OrderType.STOP_LIMIT:
            type_str = "stopLimit"
        else:
            type_str = "limit"  # Default to limit
        
        # Prepare order data
        hl_order = {
            "symbol": order.symbol,
            "side": side,
            "type": type_str,
            "quantity": str(order.quantity),
            "clientOrderId": order.id
        }
        
        # Add price for limit orders
        if order.type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and order.price is not None:
            hl_order["price"] = str(order.price)
        
        # Add stop price for stop orders
        if order.type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is not None:
            hl_order["stopPrice"] = str(order.stop_price)
        
        return hl_order
    
    def _parse_order_status(self, response_data: Dict[str, Any], order: Order) -> Dict[str, Any]:
        """
        Parse order status from Hyperliquid response.
        
        Args:
            response_data: Response data from Hyperliquid
            order: Original order
            
        Returns:
            Dictionary of status updates
        """
        updates = {}
        
        # Extract status
        status_str = response_data.get('status')
        if status_str:
            if status_str == 'filled':
                updates['status'] = OrderStatus.FILLED
            elif status_str == 'partiallyFilled':
                updates['status'] = OrderStatus.PARTIAL
            elif status_str == 'canceled':
                updates['status'] = OrderStatus.CANCELLED
            elif status_str == 'rejected':
                updates['status'] = OrderStatus.REJECTED
            elif status_str == 'expired':
                updates['status'] = OrderStatus.EXPIRED
            elif status_str == 'new':
                updates['status'] = OrderStatus.SUBMITTED
        
        # Extract filled quantity
        filled_qty = response_data.get('filledQuantity')
        if filled_qty is not None:
            updates['filled_quantity'] = float(filled_qty)
        
        # Extract average fill price
        avg_price = response_data.get('averagePrice')
        if avg_price is not None:
            updates['average_fill_price'] = float(avg_price)
        
        # Extract error
        error = response_data.get('error')
        if error:
            updates['error'] = error
        
        return updates
    
    def _sign_request(self, data: Dict[str, Any]) -> str:
        """
        Sign a request for the Hyperliquid API.
        
        Args:
            data: Request data
            
        Returns:
            Signature for the request
        """
        # Convert data to JSON string
        data_str = json.dumps(data, separators=(',', ':'))
        
        # Create signature
        timestamp = int(time.time() * 1000)
        message = f"{timestamp}{data_str}"
        
        # Sign with HMAC-SHA256
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"{timestamp}:{signature}"
