# HyperLiquid Trading System

高性能な暗号資産取引システム - HyperLiquid APIを使用した自動取引とマーケットメイキング（ETH-PERPに特化）

## 機能

- リアルタイムな市場データの表示
- 自動マーケットメイキング
- 予測モデルを用いた自動取引
- 手動取引機能
- ポジション・注文管理
- リスク管理と監視
- ドライランモードでのテスト
- ETH-PERP取引に特化した最適化

## インストール

1. リポジトリをクローン:
```bash
git clone https://github.com/yourusername/hyperliquid-trading-system.git
cd hyperliquid-trading-system
```

2. 環境セットアップスクリプトを実行:
```bash
cd project_root
python scripts/setup_environment.py
```

3. 環境変数の設定:
`.env`ファイルをプロジェクトのルートディレクトリに作成し、以下の内容を設定:
```
# Testnet Environment
HYPERLIQUID_API_KEY=your_testnet_api_key
HYPERLIQUID_API_SECRET=your_testnet_api_secret
HYPERLIQUID_API_ENDPOINT=https://api.hyperliquid-testnet.xyz
ENVIRONMENT=testnet

# Database Configuration
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=hyperliquid_trading
```

## テストネット環境のセットアップ

1. APIテストを実行して接続を確認:
```bash
cd project_root
python scripts/api_test.py
```

2. 市場データ収集を開始:
```bash
cd project_root
python scripts/collect_market_data.py --symbol ETH-PERP --duration 3600 --intervals 1h,5m,1m
```

3. 収集したデータを分析:
```bash
cd project_root
python scripts/analyze_market_data.py --symbol ETH-PERP --interval 1h --days 7
```

## 使用方法

### トレーディングエージェントの実行

1. 設定ファイルを編集:
   `project_root/config.json` を自分の戦略に合わせて編集します

2. ドライランモードでのテスト:
```bash
cd project_root
python main.py --dry-run --testnet
```

3. 実際の取引の実行:
```bash
cd project_root
python main.py --testnet
```

## システム構成

```
project_root/
├── config.json            # メイン設定ファイル
├── core/                  # コアコンポーネント
│   ├── auth.py            # 認証
│   └── config.py          # 設定管理
├── market_data/           # 市場データ処理
│   ├── data_collector.py  # データ収集
│   └── data_processor.py  # データ処理と特徴量エンジニアリング
├── models/                # 予測モデル
│   ├── lstm_model.py      # LSTMモデル
│   └── transformer_model.py # Transformerモデル
├── order_management/      # 注文管理システム
│   ├── order_types.py     # 注文タイプと構造体
│   ├── order_manager.py   # 注文ライフサイクル管理
│   ├── execution_engine.py # 実行エンジン基底クラス
│   └── hyperliquid_execution.py # HyperLiquid固有の実装
├── risk_management/       # リスク管理
│   ├── risk_monitoring/   # リスク監視
│   │   ├── risk_monitor.py # リスク指標監視
│   │   └── alert_system.py # アラートシステム
│   └── position_sizing/   # ポジションサイジング
│       └── kelly_criterion.py # ケリー基準
├── scripts/               # ユーティリティスクリプト
│   ├── api_test.py        # API接続テスト
│   ├── setup_environment.py # 環境セットアップ
│   ├── collect_market_data.py # 市場データ収集
│   └── analyze_market_data.py # 市場データ分析
├── trading_engine/        # トレーディングエンジン
│   ├── engine.py          # 基本エンジン
│   └── predictive_engine.py # 予測モデルを用いたエンジン
└── ui/                    # ユーザーインターフェース
    └── app.py
```

## ETH-PERP取引に特化した最適化

このシステムは現在、ETH-PERP取引に特化して最適化されています：

- **単一銘柄フォーカス**: システムはETH-PERPに特化し、処理とリソースを最適化
- **カスタマイズされた指標**: ETH市場の特性に合わせた技術指標の調整
- **最適化されたパラメータ**: ETH-PERPの価格変動に合わせたトレーディングパラメータ
- **効率的なデータ処理**: 単一銘柄に特化することで、データ処理とモデルトレーニングを効率化

将来的には他の銘柄にも対応予定ですが、まずはETH-PERPで安定性と収益性を確保することを優先しています。

## 新機能: データ収集と分析

### データ収集スクリプト

`scripts/collect_market_data.py`は以下の機能を提供します：

- 指定した銘柄（デフォルトはETH-PERP）の過去のローソク足データを取得
- WebSocketを使用したリアルタイム市場データのサブスクリプション
- 複数の時間枠（1分、5分、1時間など）でのデータ収集
- MongoDBへのデータ保存

### データ分析スクリプト

`scripts/analyze_market_data.py`は以下の機能を提供します：

- 収集したデータの基本的な統計分析
- 価格チャート、ボリュームチャート、各種テクニカル指標のグラフ生成
- データの完全性チェック（欠損データの特定）
- 市場傾向の分析と要約レポート生成

### 環境セットアップスクリプト

`scripts/setup_environment.py`は以下の機能を提供します：

- 必要なPythonパッケージのチェックとインストール
- MongoDBとRedisの接続確認
- .envファイルの検証と設定
- HyperLiquid APIへの接続テスト
- 必要なディレクトリの作成と初期化

## 開発者向け情報

### 環境構築

1. 開発環境のセットアップ:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. テストの実行:
```bash
python -m pytest tests/
```

### コントリビューション

1. フォークを作成
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチをプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## ライセンス

MIT License - 詳細は[LICENSE](LICENSE)ファイルを参照してください。

## トラブルシューティング

### よくある問題と解決策

1. **API認証エラー**
   - `.env`ファイルにAPI認証情報が正しく設定されているか確認
   - テストネット用のAPIキーとシークレットを使用していることを確認

2. **データ収集の問題**
   - MongoDBが実行中であることを確認
   - ネットワーク接続を確認
   - WebSocketの接続状態をログで確認

3. **実行エラー**
   - 依存関係がすべてインストールされているか確認
   - `scripts/setup_environment.py`を実行して環境を検証

4. **パフォーマンスの問題**
   - 単一銘柄（ETH-PERP）のみを処理していることを確認
   - 不要なデータ収集や処理を無効化

### サポート

問題が解決しない場合は、以下の方法でサポートを受けることができます：
- Issueを作成
- コミュニティフォーラムで質問
- ドキュメントを参照
