"""
Execution engine for submitting orders to exchanges.

This module provides the execution engine for submitting orders
to exchanges and handling responses.
"""

import logging
import asyncio
import time
import random
import json
from typing import Dict, List, Optional, Any, Callable, Coroutine, Union, Tuple
from datetime import datetime

from .order_types import (
    Order, OrderRequest, OrderCancelRequest, OrderFill, 
    OrderStatus, OrderSide, OrderType, Position
)
from .order_manager import OrderManager

# Configure logging
logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Base class for executing orders on exchanges.
    
    This class provides the basic functionality for executing orders
    on exchanges, including order submission, cancellation, and status updates.
    """
    
    def __init__(self, order_manager: OrderManager):
        """
        Initialize the execution engine.
        
        Args:
            order_manager: Order manager instance
        """
        self.order_manager = order_manager
        self.is_running = False
        self.retry_delays = [0.5, 1, 2, 5, 10, 30]  # Exponential backoff delays in seconds
        
        logger.info("Initialized execution engine")
    
    async def start(self):
        """Start the execution engine."""
        if self.is_running:
            logger.warning("Execution engine is already running")
            return
        
        self.is_running = True
        logger.info("Started execution engine")
    
    async def stop(self):
        """Stop the execution engine."""
        if not self.is_running:
            logger.warning("Execution engine is not running")
            return
        
        self.is_running = False
        logger.info("Stopped execution engine")
    
    async def submit_order(self, order_request: OrderRequest) -> Tuple[bool, Optional[Order], Optional[str]]:
        """
        Submit an order to the exchange.
        
        Args:
            order_request: Order request
            
        Returns:
            Tuple of (success, order, error_message)
        """
        # Create order from request
        order = order_request.to_order()
        
        # Add order to manager
        self.order_manager.add_order(order)
        
        # Set status to submitted
        self.order_manager.update_order(order.id, {'status': OrderStatus.SUBMITTED})
        
        # Implement exchange-specific order submission logic in subclasses
        success, error = await self._submit_order_to_exchange(order)
        
        if not success:
            # Update order status
            self.order_manager.update_order(order.id, {
                'status': OrderStatus.REJECTED,
                'error': error
            })
            return False, order, error
        
        return True, order, None
    
    async def cancel_order(self, order_id: str) -> Tuple[bool, Optional[str]]:
        """
        Cancel an order on the exchange.
        
        Args:
            order_id: Order ID
            
        Returns:
            Tuple of (success, error_message)
        """
        # Get order
        order = self.order_manager.get_order(order_id)
        if not order:
            error = f"Order {order_id} not found"
            logger.warning(error)
            return False, error
        
        # Check if order can be cancelled
        if not order.is_active():
            error = f"Order {order_id} is not active (status: {order.status.value})"
            logger.warning(error)
            return False, error
        
        # Implement exchange-specific order cancellation logic in subclasses
        success, error = await self._cancel_order_on_exchange(order)
        
        if not success:
            return False, error
        
        # Update order status
        self.order_manager.cancel_order(order_id)
        
        return True, None
    
    async def update_order_status(self, order_id: str) -> Tuple[bool, Optional[str]]:
        """
        Update the status of an order from the exchange.
        
        Args:
            order_id: Order ID
            
        Returns:
            Tuple of (success, error_message)
        """
        # Get order
        order = self.order_manager.get_order(order_id)
        if not order:
            error = f"Order {order_id} not found"
            logger.warning(error)
            return False, error
        
        # Implement exchange-specific order status update logic in subclasses
        success, status_updates, error = await self._get_order_status_from_exchange(order)
        
        if not success:
            return False, error
        
        # Update order
        self.order_manager.update_order(order_id, status_updates)
        
        return True, None
    
    async def _submit_order_to_exchange(self, order: Order) -> Tuple[bool, Optional[str]]:
        """
        Submit an order to the exchange.
        
        This method should be implemented by subclasses to handle
        exchange-specific order submission logic.
        
        Args:
            order: Order to submit
            
        Returns:
            Tuple of (success, error_message)
        """
        # This is a placeholder that should be overridden by subclasses
        logger.warning("_submit_order_to_exchange not implemented")
        return False, "Not implemented"
    
    async def _cancel_order_on_exchange(self, order: Order) -> Tuple[bool, Optional[str]]:
        """
        Cancel an order on the exchange.
        
        This method should be implemented by subclasses to handle
        exchange-specific order cancellation logic.
        
        Args:
            order: Order to cancel
            
        Returns:
            Tuple of (success, error_message)
        """
        # This is a placeholder that should be overridden by subclasses
        logger.warning("_cancel_order_on_exchange not implemented")
        return False, "Not implemented"
    
    async def _get_order_status_from_exchange(self, order: Order) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """
        Get the status of an order from the exchange.
        
        This method should be implemented by subclasses to handle
        exchange-specific order status retrieval logic.
        
        Args:
            order: Order to get status for
            
        Returns:
            Tuple of (success, status_updates, error_message)
        """
        # This is a placeholder that should be overridden by subclasses
        logger.warning("_get_order_status_from_exchange not implemented")
        return False, {}, "Not implemented"
    
    async def retry_with_backoff(self, 
                              func: Callable[..., Coroutine], 
                              *args, 
                              max_retries: int = 5,
                              **kwargs) -> Tuple[bool, Any, Optional[str]]:
        """
        Retry a function with exponential backoff.
        
        Args:
            func: Async function to retry
            *args: Arguments to pass to the function
            max_retries: Maximum number of retries
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Tuple of (success, result, error_message)
        """
        retries = 0
        last_error = None
        
        while retries <= max_retries:
            try:
                result = await func(*args, **kwargs)
                return True, result, None
            
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Error in {func.__name__} (retry {retries}/{max_retries}): {e}")
                
                if retries == max_retries:
                    break
                
                # Calculate delay with jitter
                delay = self.retry_delays[min(retries, len(self.retry_delays) - 1)]
                jitter = random.uniform(0.8, 1.2)
                sleep_time = delay * jitter
                
                logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                await asyncio.sleep(sleep_time)
                
                retries += 1
        
        return False, None, last_error
