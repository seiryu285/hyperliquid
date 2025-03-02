import { HyperLiquidAuth } from './auth';

/**
 * API呼び出しのレスポンス型
 */
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  statusCode?: number;
}

/**
 * レートリミット情報
 */
interface RateLimitInfo {
  limit: number;
  remaining: number;
  reset: number;
}

/**
 * HyperLiquid API クライアント
 * REST APIリクエストの処理を担当
 */
export class HyperLiquidApiClient {
  private auth: HyperLiquidAuth;
  private rateLimits: Map<string, RateLimitInfo> = new Map();
  private retryCount: number = 3;
  private retryDelay: number = 1000; // ミリ秒

  /**
   * コンストラクタ
   * @param auth HyperLiquidAuth インスタンス
   */
  constructor(auth: HyperLiquidAuth) {
    this.auth = auth;
  }

  /**
   * レートリミット情報を更新する
   * @param endpoint エンドポイント
   * @param headers レスポンスヘッダー
   */
  private updateRateLimits(endpoint: string, headers: Headers): void {
    const limit = headers.get('x-ratelimit-limit');
    const remaining = headers.get('x-ratelimit-remaining');
    const reset = headers.get('x-ratelimit-reset');

    if (limit && remaining && reset) {
      this.rateLimits.set(endpoint, {
        limit: parseInt(limit),
        remaining: parseInt(remaining),
        reset: parseInt(reset)
      });
    }
  }

  /**
   * レートリミットをチェックする
   * @param endpoint エンドポイント
   * @returns レートリミットに達しているかどうか
   */
  private isRateLimited(endpoint: string): boolean {
    const rateLimitInfo = this.rateLimits.get(endpoint);
    if (!rateLimitInfo) return false;

    return rateLimitInfo.remaining <= 0 && Date.now() < rateLimitInfo.reset;
  }

  /**
   * レートリミットに達した場合の待機時間を計算する
   * @param endpoint エンドポイント
   * @returns 待機時間（ミリ秒）
   */
  private getRateLimitWaitTime(endpoint: string): number {
    const rateLimitInfo = this.rateLimits.get(endpoint);
    if (!rateLimitInfo) return 0;

    return Math.max(0, rateLimitInfo.reset - Date.now());
  }

  /**
   * APIリクエストを実行する
   * @param method HTTPメソッド
   * @param endpoint エンドポイント
   * @param data リクエストデータ
   * @param requiresAuth 認証が必要かどうか
   * @returns APIレスポンス
   */
  async request<T>(
    method: string,
    endpoint: string,
    data: any = null,
    requiresAuth: boolean = true
  ): Promise<ApiResponse<T>> {
    // レートリミットチェック
    if (this.isRateLimited(endpoint)) {
      const waitTime = this.getRateLimitWaitTime(endpoint);
      console.warn(`Rate limited for ${endpoint}. Waiting ${waitTime}ms before retry.`);
      await new Promise(resolve => setTimeout(resolve, waitTime));
    }

    // リクエストURLの構築
    const url = `${this.auth.getBaseUrl()}${endpoint}`;
    
    // リクエストヘッダーの設定
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    // 認証ヘッダーの追加
    if (requiresAuth) {
      const authHeaders = this.auth.getAuthHeaders(method, endpoint, data);
      Object.assign(headers, authHeaders);
    }

    // リクエストオプションの設定
    const options: RequestInit = {
      method,
      headers,
      credentials: 'same-origin',
    };

    // リクエストボディの追加
    if (data && (method === 'POST' || method === 'PUT')) {
      options.body = JSON.stringify(data);
    }

    // リトライロジック
    let lastError: Error | null = null;
    for (let attempt = 0; attempt < this.retryCount; attempt++) {
      try {
        // リクエストの実行
        const response = await fetch(url, options);
        
        // レートリミット情報の更新
        this.updateRateLimits(endpoint, response.headers);

        // レスポンスの処理
        if (response.ok) {
          const responseData = await response.json();
          return {
            success: true,
            data: responseData as T,
            statusCode: response.status
          };
        } else {
          // エラーレスポンスの処理
          let errorMessage: string;
          try {
            const errorData = await response.json();
            errorMessage = errorData.error || errorData.message || response.statusText;
          } catch {
            errorMessage = response.statusText;
          }

          // 429エラー（レートリミット）の場合は待機してリトライ
          if (response.status === 429) {
            const retryAfter = response.headers.get('retry-after');
            const waitTime = retryAfter ? parseInt(retryAfter) * 1000 : this.retryDelay;
            console.warn(`Rate limited. Waiting ${waitTime}ms before retry.`);
            await new Promise(resolve => setTimeout(resolve, waitTime));
            continue;
          }

          return {
            success: false,
            error: errorMessage,
            statusCode: response.status
          };
        }
      } catch (error) {
        lastError = error as Error;
        
        // ネットワークエラーの場合は待機してリトライ
        if (attempt < this.retryCount - 1) {
          const waitTime = this.retryDelay * Math.pow(2, attempt);
          console.warn(`Request failed. Retrying in ${waitTime}ms...`, error);
          await new Promise(resolve => setTimeout(resolve, waitTime));
        }
      }
    }

    // すべてのリトライが失敗した場合
    return {
      success: false,
      error: lastError?.message || 'Unknown error',
    };
  }

  /**
   * GETリクエストを実行する
   * @param endpoint エンドポイント
   * @param requiresAuth 認証が必要かどうか
   * @returns APIレスポンス
   */
  async get<T>(endpoint: string, requiresAuth: boolean = true): Promise<ApiResponse<T>> {
    return this.request<T>('GET', endpoint, null, requiresAuth);
  }

  /**
   * POSTリクエストを実行する
   * @param endpoint エンドポイント
   * @param data リクエストデータ
   * @param requiresAuth 認証が必要かどうか
   * @returns APIレスポンス
   */
  async post<T>(endpoint: string, data: any, requiresAuth: boolean = true): Promise<ApiResponse<T>> {
    return this.request<T>('POST', endpoint, data, requiresAuth);
  }

  /**
   * PUTリクエストを実行する
   * @param endpoint エンドポイント
   * @param data リクエストデータ
   * @param requiresAuth 認証が必要かどうか
   * @returns APIレスポンス
   */
  async put<T>(endpoint: string, data: any, requiresAuth: boolean = true): Promise<ApiResponse<T>> {
    return this.request<T>('PUT', endpoint, data, requiresAuth);
  }

  /**
   * DELETEリクエストを実行する
   * @param endpoint エンドポイント
   * @param requiresAuth 認証が必要かどうか
   * @returns APIレスポンス
   */
  async delete<T>(endpoint: string, requiresAuth: boolean = true): Promise<ApiResponse<T>> {
    return this.request<T>('DELETE', endpoint, null, requiresAuth);
  }
}

/**
 * 環境変数から認証情報を取得してHyperLiquidApiClientインスタンスを作成する
 * @returns HyperLiquidApiClientインスタンス
 */
export const createApiClientFromEnv = (): HyperLiquidApiClient => {
  const auth = new HyperLiquidAuth(
    process.env.REACT_APP_HYPERLIQUID_API_KEY || '',
    process.env.REACT_APP_HYPERLIQUID_API_SECRET || '',
    process.env.REACT_APP_HYPERLIQUID_TESTNET !== 'false'
  );
  
  return new HyperLiquidApiClient(auth);
};
