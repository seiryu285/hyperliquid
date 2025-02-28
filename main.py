"""
Main entry point for the trading agent.

This module provides the main entry point for the trading agent,
setting up the components and starting the trading engine.
"""

import logging
import asyncio
import argparse
import os
import json
import sys
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from market_data.data_processor import DataProcessor
from order_management.order_manager import OrderManager
from order_management.hyperliquid_execution import HyperliquidExecutionEngine
from risk_management.risk_monitoring.risk_monitor import RiskMonitor
from risk_management.risk_monitoring.alert_system import alert_system, AlertType, AlertLevel
from risk_management.position_sizing.kelly_criterion import KellyCriterion
from trading_engine.predictive_engine import PredictiveEngine
from monitoring.alerts.alert_manager import alert_manager
from core.config import get_settings, settings

# Configure logging
def setup_logging(config):
    """Set up logging based on configuration."""
    log_level = getattr(logging, config.get('logging', {}).get('level', 'INFO'))
    log_file = config.get('logging', {}).get('file', 'logs/trading.log')
    max_size = config.get('logging', {}).get('max_size', 10485760)
    backup_count = config.get('logging', {}).get('backup_count', 5)
    
    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_size,
                backupCount=backup_count
            )
        ]
    )
    
    return logging.getLogger(__name__)


async def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Trading Agent')
    parser.add_argument('--dry-run', action='store_true', help='Run in dry run mode')
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration file')
    parser.add_argument('--testnet', action='store_true', help='Use testnet')
    parser.add_argument('--backtest', action='store_true', help='Run in backtest mode')
    parser.add_argument('--start-date', type=str, help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--symbol', type=str, help='Override the primary symbol to trade')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    logger = setup_logging(config)
    
    # Override config with command line arguments
    if args.dry_run:
        config['environment']['dry_run'] = True
        logger.info("Running in dry run mode")
    
    if args.testnet:
        config['environment']['mode'] = 'testnet'
        logger.info("Using testnet environment")
    
    # Get primary symbol from settings or command line
    primary_symbol = args.symbol if args.symbol else settings.PRIMARY_SYMBOL
    logger.info(f"Trading focused on primary symbol: {primary_symbol}")
    
    # Filter config to only include the primary symbol
    filter_config_for_primary_symbol(config, primary_symbol)
    
    # Log startup information
    logger.info(f"Starting trading agent with configuration from {args.config}")
    logger.info(f"Environment: {config['environment']['mode']}")
    logger.info(f"Dry run: {config['environment']['dry_run']}")
    
    # Start alert manager
    logger.info("Starting alert manager")
    alert_manager_task = asyncio.create_task(start_alert_manager(config))
    
    # Create components
    order_manager = OrderManager()
    
    # Create risk monitor
    risk_monitor = RiskMonitor(alert_system=alert_system)
    
    # Create Kelly criterion
    kelly_criterion = KellyCriterion()
    
    # Create data processor
    data_processor = DataProcessor()
    
    # Create execution engine
    execution_engine = HyperliquidExecutionEngine(
        order_manager=order_manager,
        testnet=(config['environment']['mode'] == 'testnet')
    )
    
    # Create trading engine
    trading_engine = PredictiveEngine(
        order_manager=order_manager,
        execution_engine=execution_engine,
        data_processor=data_processor,
        risk_monitor=risk_monitor,
        kelly_criterion=kelly_criterion,
        dry_run=config['environment']['dry_run']
    )
    
    # Configure trading engine
    configure_trading_engine(trading_engine, config)
    
    try:
        # Start trading engine
        await trading_engine.start()
        
        # Log initial status
        status = trading_engine.get_status()
        logger.info(f"Trading engine started: running={status['is_running']}, dry_run={status['dry_run']}")
        
        # Trigger startup alert
        alert_system.trigger_alert(
            alert_type=AlertType.SYSTEM_ERROR,
            level=AlertLevel.INFO,
            message=f"Trading agent started in {config['environment']['mode']} mode, dry_run={config['environment']['dry_run']}, symbol={primary_symbol}",
            data={"startup_time": datetime.utcnow().isoformat(), "primary_symbol": primary_symbol}
        )
        
        # Run forever
        while True:
            await asyncio.sleep(60)
            
            # Log status
            status = trading_engine.get_status()
            logger.info(f"Trading engine status: running={status['is_running']}, dry_run={status['dry_run']}")
            logger.info(f"Active positions: {len(status['positions'])}")
            logger.info(f"Active orders: {len(status['active_orders'])}")
            
            # Log primary symbol status
            primary_position = next((p for p in status['positions'] if p['symbol'] == primary_symbol), None)
            if primary_position:
                logger.info(f"Primary symbol {primary_symbol} position: {primary_position['quantity']} @ {primary_position['entry_price']}")
                logger.info(f"Unrealized PnL: {primary_position['unrealized_pnl']}, Realized PnL: {primary_position['realized_pnl']}")
            
            # Check for emergency stop conditions
            if should_emergency_stop(trading_engine, config):
                logger.warning("Emergency stop conditions met, stopping trading engine")
                alert_system.trigger_alert(
                    alert_type=AlertType.SYSTEM_ERROR,
                    level=AlertLevel.CRITICAL,
                    message="Emergency stop conditions met, stopping trading engine",
                    data={"status": status}
                )
                break
    
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
        alert_system.trigger_alert(
            alert_type=AlertType.SYSTEM_ERROR,
            level=AlertLevel.CRITICAL,
            message=f"Error in main loop: {str(e)}",
            data={"error": str(e)}
        )
    
    finally:
        # Stop trading engine
        logger.info("Stopping trading engine")
        await trading_engine.stop()
        
        # Cancel alert manager task
        logger.info("Stopping alert manager")
        alert_manager_task.cancel()
        try:
            await alert_manager_task
        except asyncio.CancelledError:
            pass
        
        logger.info("Trading agent shutdown complete")


async def start_alert_manager(config):
    """Start the alert manager."""
    try:
        port = config.get('monitoring', {}).get('prometheus', {}).get('port', 9090)
        await alert_manager.start(port=port)
    except Exception as e:
        logger.error(f"Error starting alert manager: {e}", exc_info=True)


def should_emergency_stop(trading_engine, config):
    """Check if emergency stop conditions are met."""
    status = trading_engine.get_status()
    emergency_config = config.get('risk_management', {}).get('emergency_stop', {})
    
    # Check consecutive losses
    consecutive_losses = emergency_config.get('consecutive_losses', 5)
    if status.get('consecutive_losses', 0) >= consecutive_losses:
        return True
    
    # Check drawdown
    drawdown_threshold = emergency_config.get('drawdown_threshold', 0.15)
    if status.get('drawdown', 0) >= drawdown_threshold:
        return True
    
    # Check API errors
    api_errors_threshold = emergency_config.get('api_errors_threshold', 3)
    if status.get('api_errors', 0) >= api_errors_threshold:
        return True
    
    return False


def filter_config_for_primary_symbol(config, primary_symbol):
    """
    Filter configuration to only include the primary symbol.
    
    Args:
        config: Configuration dictionary
        primary_symbol: Primary symbol to focus on
    """
    # Filter symbols
    if 'symbols' in config:
        config['symbols'] = [s for s in config['symbols'] if s.get('symbol') == primary_symbol]
        
        # Mark as primary if not already
        if config['symbols'] and not config['symbols'][0].get('is_primary'):
            config['symbols'][0]['is_primary'] = True
    
    # Filter strategies
    if 'strategies' in config:
        config['strategies'] = [
            s for s in config['strategies'] 
            if primary_symbol in s.get('symbols', [])
        ]
    
    # Set max_open_positions to 1
    if 'risk_management' in config:
        config['risk_management']['max_open_positions'] = 1
    
    return config


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Ensure required sections exist
        if 'environment' not in config:
            config['environment'] = {
                'mode': 'testnet',
                'dry_run': False,
                'api': {
                    'production': {
                        'endpoint': 'https://api.hyperliquid.xyz',
                        'ws_endpoint': 'wss://api.hyperliquid.xyz/ws'
                    },
                    'testnet': {
                        'endpoint': 'https://api.hyperliquid-testnet.xyz',
                        'ws_endpoint': 'wss://api.hyperliquid-testnet.xyz/ws'
                    }
                }
            }
        
        return config
    
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}


def configure_trading_engine(trading_engine: PredictiveEngine, config: Dict[str, Any]):
    """
    Configure trading engine from configuration.
    
    Args:
        trading_engine: Trading engine to configure
        config: Configuration dictionary
    """
    # Configure symbols
    symbols = config.get('symbols', [])
    for symbol_config in symbols:
        symbol = symbol_config.get('symbol')
        timeframes = symbol_config.get('timeframes', ['1h'])
        
        if symbol:
            trading_engine.add_symbol(symbol, timeframes)
    
    # Configure strategies
    strategies = config.get('strategies', [])
    for strategy_config in strategies:
        strategy_id = strategy_config.get('id')
        
        if strategy_id:
            trading_engine.add_strategy(strategy_id, strategy_config)
    
    # Configure risk management
    risk_config = config.get('risk_management', {})
    trading_engine.configure_risk_management(risk_config)
    
    logger.info(f"Configured trading engine with {len(symbols)} symbols and {len(strategies)} strategies")


if __name__ == '__main__':
    asyncio.run(main())
