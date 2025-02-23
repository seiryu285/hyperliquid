#!/usr/bin/env python3
"""
Test script for HyperLiquid data loader
"""

from simulator.data_loader import DataLoader

def main():
    # Initialize data loader
    loader = DataLoader()
    
    try:
        # Test market data loading
        print("Loading market data...")
        market_data = loader.load_market_data('BTC')
        print("\nMarket Data:")
        print(market_data.head())
        
        # Test historical data loading
        print("\nLoading historical data...")
        historical_data = loader.load_historical_data('BTC', days=1)
        print("\nHistorical Data:")
        print(historical_data.head())
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    main()
