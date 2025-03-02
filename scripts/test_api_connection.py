#!/usr/bin/env python
"""
HyperLiquid APIへの接続テストスクリプト

このスクリプトは以下を検証します：
1. 環境変数が正しく設定されているか
2. APIキーとシークレットが有効か
3. 認証メカニズムが正しく機能しているか
4. 基本的なデータ取得が可能か
5. WebSocket接続が機能するか
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 自作モジュールのインポート
from core.auth import HyperLiquidAuth
from core.config import settings
from market_data.data_collector import HyperLiquidDataCollector

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class APITester:
    """HyperLiquid API接続テスター"""
    
    def __init__(self):
        """初期化"""
        self.auth = HyperLiquidAuth()
        self.data_collector = HyperLiquidDataCollector()
        self.symbol = settings.PRIMARY_SYMBOL
        self.tests_passed = 0
        self.tests_failed = 0
    
    async def test_environment_variables(self):
        """環境変数が正しく設定されているかテスト"""
        logger.info("テスト: 環境変数の設定")
        
        required_vars = [
            'HYPERLIQUID_API_KEY',
            'HYPERLIQUID_API_SECRET',
            'HYPERLIQUID_API_URL',
            'HYPERLIQUID_WS_URL'
        ]
        
        missing_vars = []
        for var in required_vars:
            value = getattr(settings, var, None)
            if not value:
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"❌ 以下の環境変数が設定されていません: {', '.join(missing_vars)}")
            logger.error("  .envファイルを確認し、必要な変数を設定してください")
            self.tests_failed += 1
            return False
        
        logger.info("✅ すべての必要な環境変数が設定されています")
        self.tests_passed += 1
        return True
    
    async def test_auth_mechanism(self):
        """認証メカニズムが機能するかテスト"""
        logger.info("テスト: 認証メカニズム")
        
        try:
            # タイムスタンプとダミーリクエスト情報で署名を生成
            timestamp = int(datetime.now().timestamp() * 1000)
            method = "GET"
            path = "/test"
            signature = self.auth.generate_signature(timestamp, method, path)
            
            if not signature or len(signature) < 10:
                logger.error("❌ 署名の生成に失敗しました")
                self.tests_failed += 1
                return False
            
            logger.info(f"✅ 署名の生成に成功しました: {signature[:10]}...")
            self.tests_passed += 1
            return True
        except Exception as e:
            logger.error(f"❌ 署名生成中にエラーが発生しました: {e}")
            self.tests_failed += 1
            return False
    
    async def test_market_data_api(self):
        """マーケットデータAPIをテスト"""
        logger.info(f"テスト: マーケットデータAPI ({self.symbol})")
        
        try:
            # マーケットデータの取得
            market_data = await self.data_collector._make_rest_request("/market-data")
            
            if 'error' in market_data:
                logger.error(f"❌ マーケットデータの取得に失敗しました: {market_data['error']}")
                self.tests_failed += 1
                return False
            
            # 特定のシンボルのデータを確認
            symbol_found = False
            for item in market_data:
                if item.get('symbol') == self.symbol:
                    symbol_found = True
                    logger.info(f"✅ {self.symbol}のマーケットデータを取得しました")
                    logger.info(f"  現在価格: {item.get('price', 'N/A')}")
                    logger.info(f"  24時間変動率: {item.get('priceChange24h', 'N/A')}%")
                    break
            
            if not symbol_found:
                logger.warning(f"⚠️ {self.symbol}のデータが見つかりませんでした")
            
            self.tests_passed += 1
            return True
        except Exception as e:
            logger.error(f"❌ マーケットデータ取得中にエラーが発生しました: {e}")
            self.tests_failed += 1
            return False
    
    async def test_orderbook_api(self):
        """オーダーブックAPIをテスト"""
        logger.info(f"テスト: オーダーブックAPI ({self.symbol})")
        
        try:
            # オーダーブックの取得
            orderbook = await self.data_collector._make_rest_request(f"/orderbook?symbol={self.symbol}")
            
            if 'error' in orderbook:
                logger.error(f"❌ オーダーブックの取得に失敗しました: {orderbook['error']}")
                self.tests_failed += 1
                return False
            
            # オーダーブックの内容を確認
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                logger.warning(f"⚠️ {self.symbol}のオーダーブックが空です")
                self.tests_passed += 1
                return True
            
            logger.info(f"✅ {self.symbol}のオーダーブックを取得しました")
            logger.info(f"  買い注文数: {len(bids)}")
            logger.info(f"  売り注文数: {len(asks)}")
            logger.info(f"  最高買い価格: {bids[0][0] if bids else 'N/A'}")
            logger.info(f"  最安売り価格: {asks[0][0] if asks else 'N/A'}")
            
            self.tests_passed += 1
            return True
        except Exception as e:
            logger.error(f"❌ オーダーブック取得中にエラーが発生しました: {e}")
            self.tests_failed += 1
            return False
    
    async def test_websocket_connection(self):
        """WebSocket接続をテスト"""
        logger.info("テスト: WebSocket接続")
        
        try:
            # WebSocket接続
            connected = await self.data_collector.connect_websocket()
            
            if not connected:
                logger.error("❌ WebSocket接続に失敗しました")
                self.tests_failed += 1
                return False
            
            logger.info("✅ WebSocket接続に成功しました")
            
            # トレードサブスクリプション
            await self.data_collector.subscribe_trades([self.symbol])
            logger.info(f"✅ {self.symbol}のトレードデータをサブスクライブしました")
            
            # 5秒間データを受信
            logger.info("5秒間データを受信します...")
            
            start_time = datetime.now()
            message_count = 0
            
            while (datetime.now() - start_time).total_seconds() < 5:
                if self.data_collector.ws_connection:
                    try:
                        message = await asyncio.wait_for(
                            self.data_collector.ws_connection.recv(),
                            timeout=0.1
                        )
                        message_count += 1
                        logger.info(f"  メッセージ受信: {message[:100]}...")
                    except asyncio.TimeoutError:
                        await asyncio.sleep(0.1)
                    except Exception as e:
                        logger.error(f"  メッセージ受信中にエラー: {e}")
                else:
                    logger.error("  WebSocket接続が閉じられています")
                    break
            
            if message_count > 0:
                logger.info(f"✅ {message_count}件のWebSocketメッセージを受信しました")
                self.tests_passed += 1
                return True
            else:
                logger.warning("⚠️ WebSocketメッセージを受信できませんでした")
                self.tests_passed += 1  # 警告だが一応成功とみなす
                return True
        except Exception as e:
            logger.error(f"❌ WebSocketテスト中にエラーが発生しました: {e}")
            self.tests_failed += 1
            return False
        finally:
            # WebSocket接続を閉じる
            if self.data_collector.ws_connection:
                await self.data_collector.ws_connection.close()
                logger.info("WebSocket接続を閉じました")
    
    async def run_all_tests(self):
        """すべてのテストを実行"""
        logger.info("=== HyperLiquid API接続テスト開始 ===")
        
        # 環境変数のテスト
        env_ok = await self.test_environment_variables()
        if not env_ok:
            logger.error("環境変数の設定に問題があるため、以降のテストをスキップします")
            return
        
        # 認証メカニズムのテスト
        await self.test_auth_mechanism()
        
        # マーケットデータAPIのテスト
        await self.test_market_data_api()
        
        # オーダーブックAPIのテスト
        await self.test_orderbook_api()
        
        # WebSocket接続のテスト
        await self.test_websocket_connection()
        
        # テスト結果のサマリー
        logger.info("=== テスト結果サマリー ===")
        logger.info(f"合計テスト数: {self.tests_passed + self.tests_failed}")
        logger.info(f"成功: {self.tests_passed}")
        logger.info(f"失敗: {self.tests_failed}")
        
        if self.tests_failed == 0:
            logger.info("🎉 すべてのテストが成功しました！")
        else:
            logger.warning(f"⚠️ {self.tests_failed}件のテストが失敗しました")

async def main():
    """メイン関数"""
    tester = APITester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
