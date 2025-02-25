# HyperLiquid Trading System

高性能な暗号資産取引システム - HyperLiquid APIを使用した自動取引とマーケットメイキング

## 機能

- リアルタイムな市場データの表示
- 自動マーケットメイキング
- 手動取引機能
- ポジション・注文管理
- リスク管理

## インストール

1. リポジトリをクローン:
```bash
git clone https://github.com/yourusername/hyperliquid-trading-system.git
cd hyperliquid-trading-system
```

2. 依存関係のインストール:
```bash
pip install -r requirements.txt
```

3. 環境変数の設定:
`.env`ファイルをプロジェクトのルートディレクトリに作成し、以下の内容を設定:
```
HYPERLIQUID_API_KEY=your_api_key
HYPERLIQUID_API_SECRET=your_api_secret
```

## 使用方法

1. アプリケーションの起動:
```bash
cd project_root
streamlit run ui/app.py
```

2. ブラウザで`http://localhost:8501`を開く

3. APIキーとシークレットを設定

4. 取引設定を行い、取引を開始

## システム構成

```
project_root/
├── config/                 # 設定ファイル
│   └── market_data.yaml
├── core/                   # コアコンポーネント
│   ├── auth.py            # 認証
│   └── rest_client.py     # REST APIクライアント
├── market_data/           # 市場データ処理
│   └── websocket_client.py
├── trading/               # 取引ロジック
│   ├── order_manager.py
│   └── strategies/
│       └── simple_market_maker.py
└── ui/                    # ユーザーインターフェース
    └── app.py
```

## 注意事項

- このシステムは実際の資金を使用して取引を行います。十分なテストを行ってから使用してください。
- APIキーとシークレットは安全に管理してください。
- 取引の結果について、開発者は一切の責任を負いません。

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
