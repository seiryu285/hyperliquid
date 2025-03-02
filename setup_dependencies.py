#!/usr/bin/env python
"""
依存関係のインストールと環境設定を行うスクリプト
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """必要なPythonパッケージをインストールする"""
    requirements = [
        "aiohttp>=3.8.1",
        "websockets>=10.3",
        "pydantic>=1.9.1",
        "python-dotenv>=0.20.0",
        "motor>=3.0.0",
        "pymongo>=4.1.1",
        "redis>=4.3.4",
        "pandas>=1.4.2",
        "numpy>=1.22.4",
        "matplotlib>=3.5.2",
        "tensorflow>=2.9.1",
        "scikit-learn>=1.1.1",
        "fastapi>=0.78.0",
        "uvicorn>=0.17.6",
        "pytest>=7.1.2",
        "pytest-asyncio>=0.18.3",
    ]
    
    print("Installing Python dependencies...")
    for package in requirements:
        print(f"Installing {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("All dependencies installed successfully!")

def check_external_services():
    """外部サービス（MongoDB、Redis）の状態を確認する"""
    # MongoDB
    try:
        import pymongo
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
        client.server_info()  # Will raise an exception if cannot connect
        print("✅ MongoDB is running and accessible")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("Please make sure MongoDB is installed and running on localhost:27017")
    
    # Redis
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()  # Will raise an exception if cannot connect
        print("✅ Redis is running and accessible")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("Please make sure Redis is installed and running on localhost:6379")

def create_env_file_template():
    """
    .envファイルのテンプレートを作成する
    注意: 既存の.envファイルは上書きしない
    """
    env_path = Path(".env")
    
    if env_path.exists():
        print("⚠️ .env file already exists. Not overwriting.")
        return
    
    env_content = """# Hyperliquid API Configuration - Testnet Environment
HYPERLIQUID_API_KEY=your_testnet_api_key
HYPERLIQUID_API_SECRET=your_testnet_api_secret
HYPERLIQUID_API_ENDPOINT=https://api.hyperliquid-testnet.xyz
HYPERLIQUID_WS_URL=wss://api.hyperliquid-testnet.xyz/ws

# Environment Selection
ENVIRONMENT=testnet

# Database Configuration
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=hyperliquid_trading

# Redis Configuration (if needed)
REDIS_URL=redis://localhost:6379
"""
    
    with open(env_path, "w") as f:
        f.write(env_content)
    
    print("✅ Created .env file template")
    print("⚠️ Please edit the .env file and add your actual API credentials")

def main():
    """メイン関数"""
    print("Setting up HyperLiquid Agent environment...")
    
    # 依存関係のインストール
    install_dependencies()
    
    # 外部サービスの確認
    check_external_services()
    
    # .envファイルテンプレートの作成
    create_env_file_template()
    
    print("\nSetup completed!")
    print("Next steps:")
    print("1. Edit the .env file with your actual API credentials")
    print("2. Start MongoDB and Redis if they're not running")
    print("3. Run the API connection test script: python scripts/test_api_connection.py")

if __name__ == "__main__":
    main()
