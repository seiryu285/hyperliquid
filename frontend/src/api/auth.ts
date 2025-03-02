import CryptoJS from 'crypto-js';

/**
 * HyperLiquid API認証クラス
 * 署名生成と認証ヘッダーの作成を担当
 */
export class HyperLiquidAuth {
  private apiKey: string;
  private apiSecret: string;
  private testnet: boolean;

  /**
   * コンストラクタ
   * @param apiKey API Key
   * @param apiSecret API Secret
   * @param testnet テストネットを使用するかどうか
   */
  constructor(apiKey: string, apiSecret: string, testnet: boolean = true) {
    this.apiKey = apiKey;
    this.apiSecret = apiSecret;
    this.testnet = testnet;
  }

  /**
   * 署名を生成する
   * @param timestamp タイムスタンプ（ミリ秒）
   * @param method HTTPメソッド
   * @param path リクエストパス
   * @param data リクエストデータ
   * @returns 生成された署名
   */
  generateSignature(timestamp: number, method: string, path: string, data: any = ''): string {
    // メッセージの構築
    const message = `${timestamp}${method}${path}${typeof data === 'string' ? data : JSON.stringify(data)}`;
    
    // HMAC-SHA256を使用して署名を生成
    const signature = CryptoJS.HmacSHA256(message, this.apiSecret).toString(CryptoJS.enc.Hex);
    
    return signature;
  }

  /**
   * 認証ヘッダーを生成する
   * @param method HTTPメソッド
   * @param path リクエストパス
   * @param data リクエストデータ
   * @returns 認証ヘッダーオブジェクト
   */
  getAuthHeaders(method: string, path: string, data: any = ''): Record<string, string> {
    const timestamp = Date.now();
    const signature = this.generateSignature(timestamp, method, path, data);
    
    return {
      'hl-api-key': this.apiKey,
      'hl-api-timestamp': timestamp.toString(),
      'hl-api-signature': signature,
    };
  }

  /**
   * WebSocket認証メッセージを生成する
   * @returns WebSocket認証メッセージ
   */
  getWebSocketAuthMessage(): any {
    const timestamp = Date.now();
    const signature = this.generateSignature(timestamp, 'GET', '/ws', '');
    
    return {
      op: 'auth',
      args: [
        this.apiKey,
        timestamp.toString(),
        signature
      ]
    };
  }

  /**
   * APIのベースURLを取得する
   * @returns APIのベースURL
   */
  getBaseUrl(): string {
    return this.testnet 
      ? 'https://api-testnet.hyperliquid.xyz' 
      : 'https://api.hyperliquid.xyz';
  }

  /**
   * WebSocketのURLを取得する
   * @returns WebSocketのURL
   */
  getWebSocketUrl(): string {
    return this.testnet 
      ? 'wss://api-testnet.hyperliquid.xyz/ws' 
      : 'wss://api.hyperliquid.xyz/ws';
  }
}

/**
 * 環境変数から認証情報を取得してHyperLiquidAuthインスタンスを作成する
 * @returns HyperLiquidAuthインスタンス
 */
export const createAuthFromEnv = (): HyperLiquidAuth => {
  const apiKey = process.env.REACT_APP_HYPERLIQUID_API_KEY || '';
  const apiSecret = process.env.REACT_APP_HYPERLIQUID_API_SECRET || '';
  const testnet = process.env.REACT_APP_HYPERLIQUID_TESTNET !== 'false';
  
  if (!apiKey || !apiSecret) {
    console.warn('API Key or Secret not found in environment variables');
  }
  
  return new HyperLiquidAuth(apiKey, apiSecret, testnet);
};
