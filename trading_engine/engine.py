"""
Trading engine for executing trading strategies.

This module provides the base TradingEngine class for executing
trading strategies using the order management system.
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Coroutine, Union, Tuple
from datetime import datetime

from market_data.data_processor import DataProcessor
from order_management.order_manager import OrderManager
from order_management.execution_engine import ExecutionEngine
from order_management.order_types import (
    OrderSide, OrderType, OrderStatus, OrderTimeInForce,
    Order, OrderFill, Position, OrderRequest, OrderCancelRequest
)
from risk_management.risk_monitoring.risk_monitor import RiskMonitor
from risk_management.position_sizing.kelly_criterion import KellyCriterion
from core.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()


class TradingEngine:
    """
    Base class for executing trading strategies.
    
    This class provides the basic functionality for executing trading
    strategies using the order management system.
    """
    
    def __init__(self, 
                order_manager: OrderManager,
                execution_engine: ExecutionEngine,
                data_processor: DataProcessor,
                risk_monitor: Optional[RiskMonitor] = None,
                kelly_criterion: Optional[KellyCriterion] = None,
                dry_run: bool = True):
        """
        Initialize the trading engine.
        
        Args:
            order_manager: Order manager instance
            execution_engine: Execution engine instance
            data_processor: Data processor instance
            risk_monitor: Risk monitor instance
            kelly_criterion: Kelly criterion instance
            dry_run: Whether to run in dry run mode
        """
        self.order_manager = order_manager
        self.execution_engine = execution_engine
        self.data_processor = data_processor
        self.risk_monitor = risk_monitor
        self.kelly_criterion = kelly_criterion
        self.dry_run = dry_run
        
        self.is_running = False
        self.trading_symbols = []
        self.trading_timeframes = []
        self.strategy_configs = {}
        
        logger.info(f"Initialized trading engine (dry_run={dry_run})")
    
    async def start(self):
        """Start the trading engine."""
        if self.is_running:
            logger.warning("Trading engine is already running")
            return
        
        # Start execution engine
        await self.execution_engine.start()
        
        self.is_running = True
        logger.info("Started trading engine")
        
        # Start main loop
        asyncio.create_task(self._main_loop())
    
    async def stop(self):
        """Stop the trading engine."""
        if not self.is_running:
            logger.warning("Trading engine is not running")
            return
        
        self.is_running = False
        
        # Stop execution engine
        await self.execution_engine.stop()
        
        logger.info("Stopped trading engine")
    
    def add_symbol(self, symbol: str, timeframes: List[str]):
        """
        Add a symbol to trade.
        
        Args:
            symbol: Symbol to trade
            timeframes: Timeframes to use for this symbol
        """
        if symbol not in self.trading_symbols:
            self.trading_symbols.append(symbol)
            self.trading_timeframes.append(timeframes)
            logger.info(f"Added symbol {symbol} with timeframes {timeframes}")
    
    def add_strategy(self, strategy_id: str, strategy_config: Dict[str, Any]):
        """
        Add a strategy configuration.
        
        Args:
            strategy_id: Strategy ID
            strategy_config: Strategy configuration
        """
        self.strategy_configs[strategy_id] = strategy_config
        logger.info(f"Added strategy {strategy_id}")
    
    async def _main_loop(self):
        """Main trading loop."""
        logger.info("Starting main trading loop")
        
        while self.is_running:
            try:
                # Update market data
                await self._update_market_data()
                
                # Generate signals
                signals = await self._generate_signals()
                
                # Execute signals
                await self._execute_signals(signals)
                
                # Update positions
                await self._update_positions()
                
                # Update risk metrics
                await self._update_risk_metrics()
                
                # Sleep before next iteration
                await asyncio.sleep(settings.TRADING_ENGINE_LOOP_INTERVAL)
            
            except Exception as e:
                logger.error(f"Error in main trading loop: {e}")
                await asyncio.sleep(5)  # Sleep before retrying
    
    async def _update_market_data(self):
        """Update market data."""
        logger.debug("Updating market data")
        
        for i, symbol in enumerate(self.trading_symbols):
            timeframes = self.trading_timeframes[i]
            
            for timeframe in timeframes:
                try:
                    # Fetch OHLCV data
                    data = await self.data_processor.fetch_ohlcv_data(
                        symbol=symbol,
                        timeframe=timeframe
                    )
                    
                    # Add technical indicators
                    data = self.data_processor.add_technical_indicators(data)
                    
                    logger.debug(f"Updated market data for {symbol} {timeframe}")
                
                except Exception as e:
                    logger.error(f"Error updating market data for {symbol} {timeframe}: {e}")
    
    async def _generate_signals(self) -> List[Dict[str, Any]]:
        """
        Generate trading signals.
        
        Returns:
            List of trading signals
        """
        logger.debug("Generating signals")
        
        signals = []
        
        for strategy_id, config in self.strategy_configs.items():
            try:
                # Get strategy type
                strategy_type = config.get('type')
                if not strategy_type:
                    logger.warning(f"No type specified for strategy {strategy_id}")
                    continue
                
                # Get strategy parameters
                params = config.get('parameters', {})
                
                # Get symbols for this strategy
                symbols = config.get('symbols', self.trading_symbols)
                
                # Get timeframes for this strategy
                timeframes = config.get('timeframes', ['1h'])
                
                for symbol in symbols:
                    for timeframe in timeframes:
                        # Generate signals for this symbol and timeframe
                        strategy_signals = await self._generate_strategy_signals(
                            strategy_type=strategy_type,
                            symbol=symbol,
                            timeframe=timeframe,
                            params=params
                        )
                        
                        # Add signals to list
                        signals.extend(strategy_signals)
            
            except Exception as e:
                logger.error(f"Error generating signals for strategy {strategy_id}: {e}")
        
        logger.info(f"Generated {len(signals)} signals")
        return signals
    
    async def _generate_strategy_signals(self, 
                                       strategy_type: str,
                                       symbol: str,
                                       timeframe: str,
                                       params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate signals for a specific strategy, symbol, and timeframe.
        
        Args:
            strategy_type: Type of strategy
            symbol: Symbol
            timeframe: Timeframe
            params: Strategy parameters
            
        Returns:
            List of signals
        """
        # This method should be implemented by subclasses
        logger.warning("_generate_strategy_signals not implemented")
        return []
    
    async def _execute_signals(self, signals: List[Dict[str, Any]]):
        """
        Execute trading signals.
        
        Args:
            signals: List of trading signals
        """
        logger.debug(f"Executing {len(signals)} signals")
        
        for signal in signals:
            try:
                # Extract signal information
                symbol = signal.get('symbol')
                side = signal.get('side')
                quantity = signal.get('quantity')
                price = signal.get('price')
                order_type = signal.get('order_type', 'market')
                
                if not symbol or not side or not quantity:
                    logger.warning(f"Invalid signal: {signal}")
                    continue
                
                # Check risk limits
                if self.risk_monitor and not self.risk_monitor.check_trade_risk(symbol, side, quantity, price):
                    logger.warning(f"Trade rejected by risk monitor: {signal}")
                    continue
                
                # Apply position sizing if available
                if self.kelly_criterion and 'win_rate' in signal and 'risk_reward' in signal:
                    win_rate = signal['win_rate']
                    risk_reward = signal['risk_reward']
                    
                    # Calculate Kelly fraction
                    kelly_fraction = self.kelly_criterion.calculate_kelly_fraction(win_rate, risk_reward)
                    
                    # Apply Kelly fraction to quantity
                    quantity = quantity * kelly_fraction
                    
                    logger.info(f"Applied Kelly sizing: {kelly_fraction:.2f} * {quantity} = {quantity * kelly_fraction}")
                
                # Create order request
                order_request = OrderRequest(
                    symbol=symbol,
                    side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
                    type=OrderType.MARKET if order_type.lower() == 'market' else OrderType.LIMIT,
                    quantity=abs(quantity),
                    price=price
                )
                
                # Execute order
                if self.dry_run:
                    # In dry run mode, just log the order
                    logger.info(f"[DRY RUN] Would submit order: {order_request}")
                else:
                    # Submit order to exchange
                    success, order, error = await self.execution_engine.submit_order(order_request)
                    
                    if success:
                        logger.info(f"Submitted order: {order.id}")
                    else:
                        logger.error(f"Error submitting order: {error}")
            
            except Exception as e:
                logger.error(f"Error executing signal: {e}")
    
    async def _update_positions(self):
        """Update positions from the exchange."""
        logger.debug("Updating positions")
        
        try:
            # In dry run mode, skip position updates
            if self.dry_run:
                return
            
            # Update positions from exchange
            await self.execution_engine.update_positions()
            
            logger.debug("Updated positions")
        
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    async def _update_risk_metrics(self):
        """Update risk metrics."""
        logger.debug("Updating risk metrics")
        
        try:
            # Skip if no risk monitor
            if not self.risk_monitor:
                return
            
            # Get positions
            positions = self.order_manager.get_all_positions()
            
            # Update risk metrics
            self.risk_monitor.update_positions(positions)
            
            # Check risk limits
            risk_alerts = self.risk_monitor.check_risk_limits()
            
            # Handle risk alerts
            for alert in risk_alerts:
                logger.warning(f"Risk alert: {alert}")
                
                # Take action based on alert
                if alert.get('action') == 'reduce_position':
                    symbol = alert.get('symbol')
                    reduction_pct = alert.get('reduction_pct', 0.5)
                    
                    # Get current position
                    position = self.order_manager.get_position(symbol)
                    
                    # Calculate reduction amount
                    reduction_qty = position.quantity * reduction_pct
                    
                    # Create order to reduce position
                    if abs(reduction_qty) > 0:
                        side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
                        
                        order_request = OrderRequest(
                            symbol=symbol,
                            side=side,
                            type=OrderType.MARKET,
                            quantity=abs(reduction_qty)
                        )
                        
                        # Execute order
                        if self.dry_run:
                            logger.info(f"[DRY RUN] Would reduce position: {order_request}")
                        else:
                            success, order, error = await self.execution_engine.submit_order(order_request)
                            
                            if success:
                                logger.info(f"Reduced position: {order.id}")
                            else:
                                logger.error(f"Error reducing position: {error}")
            
            logger.debug("Updated risk metrics")
        
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the trading engine.
        
        Returns:
            Dictionary with status information
        """
        return {
            'is_running': self.is_running,
            'dry_run': self.dry_run,
            'trading_symbols': self.trading_symbols,
            'trading_timeframes': self.trading_timeframes,
            'strategy_configs': self.strategy_configs,
            'positions': [pos.dict() for pos in self.order_manager.get_all_positions()],
            'active_orders': [order.dict() for order in self.order_manager.get_active_orders()]
        }
