#!/usr/bin/env python
"""
Hyperliquid Trading Environment Setup Script

This script helps set up the environment for the Hyperliquid trading agent:
1. Checks and installs required Python packages
2. Verifies MongoDB and Redis connections
3. Validates the .env file configuration
4. Tests API connectivity to Hyperliquid testnet
5. Creates necessary directories and initializes database collections

Usage:
    python setup_environment.py [--force-install] [--skip-db-check]
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
import importlib.util
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Required packages
REQUIRED_PACKAGES = [
    'aiohttp',
    'websockets',
    'pymongo',
    'motor',
    'pandas',
    'numpy',
    'python-dotenv',
    'matplotlib',
    'redis',
    'pydantic',
    'pytest',
    'requests',
    'celery',
    'scikit-learn',
    'tensorflow',
    'hmac',
    'hashlib',
]

def check_package(package_name):
    """Check if a package is installed."""
    try:
        importlib.util.find_spec(package_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """Install a package using pip."""
    logger.info(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package_name}: {e}")
        return False

def check_and_install_packages(force_install=False):
    """Check and install required packages."""
    logger.info("Checking required Python packages...")
    
    missing_packages = []
    for package in REQUIRED_PACKAGES:
        if not check_package(package) or force_install:
            missing_packages.append(package)
    
    if missing_packages:
        logger.info(f"Missing packages: {', '.join(missing_packages)}")
        
        for package in missing_packages:
            install_package(package)
        
        logger.info("Package installation completed")
    else:
        logger.info("All required packages are already installed")

def check_mongodb_connection():
    """Check MongoDB connection."""
    logger.info("Checking MongoDB connection...")
    
    try:
        import pymongo
        from pymongo import MongoClient
        
        # Add project root to path
        project_root = Path(__file__).parent.parent
        sys.path.append(str(project_root))
        
        from dotenv import load_dotenv
        load_dotenv(project_root / '.env')
        
        # Get MongoDB URI from environment
        mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
        
        # Try to connect
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        server_info = client.server_info()
        
        logger.info(f"Successfully connected to MongoDB (version: {server_info.get('version')})")
        
        # Create database and collections if they don't exist
        db = client.get_database('hyperliquid_trading')
        
        # Create collections
        collections = ['trades', 'orderbook', 'candles', 'ticker', 'positions', 'orders']
        for collection_name in collections:
            if collection_name not in db.list_collection_names():
                db.create_collection(collection_name)
                logger.info(f"Created collection: {collection_name}")
        
        # Create indexes
        db.trades.create_index([("symbol", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)])
        db.orderbook.create_index([("symbol", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)])
        db.candles.create_index([("symbol", pymongo.ASCENDING), ("interval", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)])
        db.ticker.create_index([("symbol", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)])
        
        logger.info("MongoDB collections and indexes created successfully")
        
        return True
    except Exception as e:
        logger.error(f"MongoDB connection error: {e}")
        logger.error("Please make sure MongoDB is installed and running")
        return False

def check_redis_connection():
    """Check Redis connection."""
    logger.info("Checking Redis connection...")
    
    try:
        import redis
        
        # Add project root to path
        project_root = Path(__file__).parent.parent
        sys.path.append(str(project_root))
        
        from dotenv import load_dotenv
        load_dotenv(project_root / '.env')
        
        # Get Redis URI from environment
        redis_uri = os.getenv('REDIS_URL', 'redis://localhost:6379')
        
        # Try to connect
        r = redis.from_url(redis_uri)
        r.ping()
        
        logger.info("Successfully connected to Redis")
        return True
    except Exception as e:
        logger.error(f"Redis connection error: {e}")
        logger.error("Please make sure Redis is installed and running")
        return False

def validate_env_file():
    """Validate .env file configuration."""
    logger.info("Validating .env file configuration...")
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    
    from dotenv import load_dotenv
    load_dotenv(project_root / '.env')
    
    # Required environment variables
    required_vars = [
        'HYPERLIQUID_API_KEY',
        'HYPERLIQUID_API_SECRET',
        'ENVIRONMENT',
        'MONGODB_URI',
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.error("Please update your .env file with the required variables")
        
        # Check if .env file exists
        if not os.path.exists(project_root / '.env'):
            logger.error(".env file not found. Creating from .env.example...")
            
            # Copy .env.example to .env if it exists
            if os.path.exists(project_root / '.env.example'):
                with open(project_root / '.env.example', 'r') as src:
                    with open(project_root / '.env', 'w') as dst:
                        dst.write(src.read())
                logger.info("Created .env file from .env.example")
                logger.info("Please update the .env file with your API credentials")
            else:
                logger.error(".env.example file not found. Please create a .env file manually")
        
        return False
    
    logger.info("Environment variables validated successfully")
    return True

def test_api_connectivity():
    """Test API connectivity to Hyperliquid testnet."""
    logger.info("Testing API connectivity to Hyperliquid testnet...")
    
    try:
        import aiohttp
        import asyncio
        
        # Add project root to path
        project_root = Path(__file__).parent.parent
        sys.path.append(str(project_root))
        
        from dotenv import load_dotenv
        load_dotenv(project_root / '.env')
        
        from core.auth import HyperLiquidAuth
        
        # Get API credentials
        api_key = os.getenv('HYPERLIQUID_API_KEY', '')
        api_secret = os.getenv('HYPERLIQUID_API_SECRET', '')
        
        if not api_key or not api_secret:
            logger.error("API credentials not found in .env file")
            return False
        
        # Test API connectivity
        async def test_api():
            auth = HyperLiquidAuth()
            api_url = "https://api.hyperliquid-testnet.xyz"
            
            # Test public endpoint
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{api_url}/info") as response:
                    if response.status == 200:
                        logger.info("Successfully connected to Hyperliquid testnet API (public endpoint)")
                    else:
                        logger.error(f"Failed to connect to public API endpoint: {response.status}")
                        return False
            
            # Test authenticated endpoint
            timestamp = int(time.time() * 1000)
            signature = auth.generate_signature(timestamp, "GET", "/user/info", None)
            
            headers = {
                "HL-API-KEY": api_key,
                "HL-SIGNATURE": signature,
                "HL-TIMESTAMP": str(timestamp)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{api_url}/user/info", headers=headers) as response:
                    if response.status == 200:
                        logger.info("Successfully authenticated with Hyperliquid testnet API")
                        return True
                    else:
                        logger.error(f"Failed to authenticate with API: {response.status}")
                        logger.error(await response.text())
                        return False
        
        return asyncio.run(test_api())
    
    except Exception as e:
        logger.error(f"API connectivity test error: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    logger.info("Creating necessary directories...")
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    
    # Directories to create
    directories = [
        'logs',
        'data',
        'models',
        'analysis_output',
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            logger.info(f"Created directory: {directory}")
        else:
            logger.info(f"Directory already exists: {directory}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Setup Hyperliquid trading environment')
    parser.add_argument('--force-install', action='store_true', help='Force reinstallation of packages')
    parser.add_argument('--skip-db-check', action='store_true', help='Skip database connection checks')
    
    args = parser.parse_args()
    
    logger.info("Starting environment setup...")
    
    # Check and install packages
    check_and_install_packages(args.force_install)
    
    # Create directories
    create_directories()
    
    # Validate .env file
    env_valid = validate_env_file()
    
    # Check database connections
    db_valid = True
    if not args.skip_db_check:
        mongo_valid = check_mongodb_connection()
        redis_valid = check_redis_connection()
        db_valid = mongo_valid and redis_valid
    
    # Test API connectivity
    api_valid = test_api_connectivity()
    
    # Summary
    logger.info("\n=== Setup Summary ===")
    logger.info(f"Environment variables: {'✓' if env_valid else '✗'}")
    logger.info(f"Database connections: {'✓' if db_valid else '✗'}")
    logger.info(f"API connectivity: {'✓' if api_valid else '✗'}")
    
    if env_valid and db_valid and api_valid:
        logger.info("\nEnvironment setup completed successfully!")
        logger.info("You can now run the trading agent")
    else:
        logger.error("\nEnvironment setup completed with errors")
        logger.error("Please fix the issues before running the trading agent")
    
    return env_valid and db_valid and api_valid

if __name__ == '__main__':
    main()
