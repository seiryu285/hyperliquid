import { HyperLiquidAuth } from './auth';

/**
 * WebSocketメッセージハンドラの型
 */
export type WebSocketMessageHandler = (data: any) => void;

/**
 * WebSocketの状態
 */
export enum WebSocketState {
  CONNECTING = 'connecting',
  OPEN = 'open',
  AUTHENTICATED = 'authenticated',
  CLOSED = 'closed',
  ERROR = 'error'
}

/**
 * WebSocketの接続オプション
 */
export interface WebSocketOptions {
  reconnectOnClose?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  pingInterval?: number;
  messageBufferSize?: number; // メッセージバッファサイズ
  batchProcessing?: boolean;  // メッセージの一括処理を有効にするか
  batchInterval?: number;     // メッセージの一括処理間隔（ミリ秒）
}

/**
 * WebSocketの統計情報
 */
export interface WebSocketStats {
  latency: number | null;      // レイテンシ（ミリ秒）
  messageRate: number;         // 1秒あたりのメッセージ数
  bytesReceived: number;       // 受信バイト数
  reconnectCount: number;      // 再接続回数
  lastMessageTime: number | null; // 最後にメッセージを受信した時間
}

/**
 * HyperLiquid WebSocketクライアント
 * リアルタイムデータの取得を担当
 */
export class HyperLiquidWebSocketClient {
  private auth: HyperLiquidAuth;
  private ws: WebSocket | null = null;
  private state: WebSocketState = WebSocketState.CLOSED;
  private messageHandlers: Map<string, WebSocketMessageHandler[]> = new Map();
  private reconnectAttempts: number = 0;
  private pingIntervalId: number | null = null;
  private subscriptions: Set<string> = new Set();
  private messageBuffer: any[] = [];
  private processingIntervalId: number | null = null;
  private lastPingTime: number | null = null;
  private lastPongTime: number | null = null;
  
  // 統計情報
  private stats: WebSocketStats = {
    latency: null,
    messageRate: 0,
    bytesReceived: 0,
    reconnectCount: 0,
    lastMessageTime: null
  };
  
  // メッセージレート計算用
  private messageCount: number = 0;
  private messageRateIntervalId: number | null = null;
  
  private options: WebSocketOptions = {
    reconnectOnClose: true,
    reconnectInterval: 1000,
    maxReconnectAttempts: 10,
    pingInterval: 30000, // 30秒ごとにpingを送信
    messageBufferSize: 100, // メッセージバッファサイズ
    batchProcessing: true,  // メッセージの一括処理を有効にする
    batchInterval: 16       // 約60FPSに相当（16.67ms）
  };

  /**
   * コンストラクタ
   * @param auth HyperLiquidAuth インスタンス
   * @param options WebSocketオプション
   */
  constructor(auth: HyperLiquidAuth, options?: Partial<WebSocketOptions>) {
    this.auth = auth;
    this.options = { ...this.options, ...options };
  }

  /**
   * WebSocketの状態を取得する
   * @returns 現在のWebSocketの状態
   */
  getState(): WebSocketState {
    return this.state;
  }

  /**
   * WebSocketの統計情報を取得する
   * @returns 統計情報
   */
  getStats(): WebSocketStats {
    return { ...this.stats };
  }

  /**
   * WebSocketに接続する
   * @returns 接続が成功したかどうか
   */
  async connect(): Promise<boolean> {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      console.log('WebSocket is already connected or connecting');
      return true;
    }

    return new Promise((resolve) => {
      try {
        this.state = WebSocketState.CONNECTING;
        this.ws = new WebSocket(this.auth.getWebSocketUrl());

        // バイナリデータ形式を指定（パフォーマンス向上）
        if (this.ws.binaryType) {
          this.ws.binaryType = 'arraybuffer';
        }

        this.ws.onopen = () => {
          console.log('WebSocket connected');
          this.state = WebSocketState.OPEN;
          this.reconnectAttempts = 0;
          
          // 認証
          this.authenticate();
          
          // Pingの定期送信を開始
          this.startPingInterval();
          
          // メッセージレート計測を開始
          this.startMessageRateCalculation();
          
          // メッセージの一括処理を開始
          if (this.options.batchProcessing) {
            this.startBatchProcessing();
          }
          
          // 以前のサブスクリプションを再登録
          this.resubscribe();
          
          resolve(true);
        };

        this.ws.onmessage = (event) => {
          // 受信バイト数を更新
          if (typeof event.data === 'string') {
            this.stats.bytesReceived += event.data.length;
          } else if (event.data instanceof ArrayBuffer) {
            this.stats.bytesReceived += event.data.byteLength;
          }
          
          // メッセージカウントを更新
          this.messageCount++;
          
          // 最後のメッセージ時間を更新
          this.stats.lastMessageTime = Date.now();
          
          if (this.options.batchProcessing) {
            // バッファにメッセージを追加
            this.addToMessageBuffer(event);
          } else {
            // 即時処理
            this.handleMessage(event);
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.state = WebSocketState.ERROR;
          resolve(false);
        };

        this.ws.onclose = () => {
          console.log('WebSocket closed');
          this.state = WebSocketState.CLOSED;
          this.clearPingInterval();
          this.clearMessageRateCalculation();
          this.clearBatchProcessing();
          
          // 再接続回数を更新
          this.stats.reconnectCount++;
          
          // 自動再接続
          if (this.options.reconnectOnClose && this.reconnectAttempts < (this.options.maxReconnectAttempts || 10)) {
            this.reconnectAttempts++;
            const reconnectDelay = this.options.reconnectInterval || 1000;
            console.log(`Reconnecting in ${reconnectDelay}ms (attempt ${this.reconnectAttempts})`);
            setTimeout(() => this.connect(), reconnectDelay);
          }
          
          resolve(false);
        };
      } catch (error) {
        console.error('Error connecting to WebSocket:', error);
        this.state = WebSocketState.ERROR;
        resolve(false);
      }
    });
  }

  /**
   * WebSocketを切断する
   */
  disconnect(): void {
    this.clearPingInterval();
    this.clearMessageRateCalculation();
    this.clearBatchProcessing();
    this.subscriptions.clear();
    
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    
    this.state = WebSocketState.CLOSED;
  }

  /**
   * WebSocketに認証メッセージを送信する
   */
  private authenticate(): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.error('WebSocket is not open');
      return;
    }

    const authMessage = this.auth.getWebSocketAuthMessage();
    this.ws.send(JSON.stringify(authMessage));
    
    // 認証メッセージのハンドラを一時的に登録
    const authHandler = (data: any) => {
      if (data.op === 'auth' && data.success) {
        console.log('WebSocket authenticated successfully');
        this.state = WebSocketState.AUTHENTICATED;
        
        // ハンドラを削除
        this.removeMessageHandler('auth', authHandler);
      } else if (data.op === 'auth') {
        console.error('WebSocket authentication failed:', data.message);
        this.state = WebSocketState.ERROR;
        
        // ハンドラを削除
        this.removeMessageHandler('auth', authHandler);
      }
    };
    
    this.addMessageHandler('auth', authHandler);
  }

  /**
   * Pingの定期送信を開始する
   */
  private startPingInterval(): void {
    this.clearPingInterval();
    
    if (this.options.pingInterval) {
      this.pingIntervalId = window.setInterval(() => {
        this.sendPing();
      }, this.options.pingInterval);
    }
  }

  /**
   * Pingの定期送信を停止する
   */
  private clearPingInterval(): void {
    if (this.pingIntervalId !== null) {
      clearInterval(this.pingIntervalId);
      this.pingIntervalId = null;
    }
  }

  /**
   * メッセージレート計算を開始する
   */
  private startMessageRateCalculation(): void {
    this.clearMessageRateCalculation();
    
    // 1秒ごとにメッセージレートを計算
    this.messageRateIntervalId = window.setInterval(() => {
      this.stats.messageRate = this.messageCount;
      this.messageCount = 0;
    }, 1000);
  }

  /**
   * メッセージレート計算を停止する
   */
  private clearMessageRateCalculation(): void {
    if (this.messageRateIntervalId !== null) {
      clearInterval(this.messageRateIntervalId);
      this.messageRateIntervalId = null;
    }
  }

  /**
   * メッセージの一括処理を開始する
   */
  private startBatchProcessing(): void {
    this.clearBatchProcessing();
    
    if (this.options.batchProcessing && this.options.batchInterval) {
      this.processingIntervalId = window.setInterval(() => {
        this.processMessageBuffer();
      }, this.options.batchInterval);
    }
  }

  /**
   * メッセージの一括処理を停止する
   */
  private clearBatchProcessing(): void {
    if (this.processingIntervalId !== null) {
      clearInterval(this.processingIntervalId);
      this.processingIntervalId = null;
    }
  }

  /**
   * メッセージバッファにメッセージを追加する
   * @param event WebSocketのメッセージイベント
   */
  private addToMessageBuffer(event: MessageEvent): void {
    // バッファサイズを超えた場合は古いメッセージを削除
    if (this.messageBuffer.length >= (this.options.messageBufferSize || 100)) {
      this.messageBuffer.shift();
    }
    
    this.messageBuffer.push(event);
  }

  /**
   * メッセージバッファを処理する
   */
  private processMessageBuffer(): void {
    if (this.messageBuffer.length === 0) {
      return;
    }
    
    // バッファ内のすべてのメッセージを処理
    const messages = [...this.messageBuffer];
    this.messageBuffer = [];
    
    // メッセージを処理
    for (const event of messages) {
      this.handleMessage(event);
    }
  }

  /**
   * Pingメッセージを送信する
   */
  private sendPing(): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }
    
    this.lastPingTime = Date.now();
    
    const pingMessage = {
      op: 'ping',
      timestamp: this.lastPingTime
    };
    
    this.ws.send(JSON.stringify(pingMessage));
    
    // Pongメッセージのハンドラを一時的に登録
    const pongHandler = (data: any) => {
      if (data.op === 'pong') {
        this.lastPongTime = Date.now();
        
        // レイテンシを計算（往復時間）
        if (this.lastPingTime) {
          this.stats.latency = this.lastPongTime - this.lastPingTime;
        }
        
        // ハンドラを削除
        this.removeMessageHandler('pong', pongHandler);
      }
    };
    
    this.addMessageHandler('pong', pongHandler);
  }

  /**
   * 以前のサブスクリプションを再登録する
   */
  private resubscribe(): void {
    if (this.subscriptions.size === 0) {
      return;
    }
    
    console.log('Resubscribing to channels:', Array.from(this.subscriptions));
    
    for (const channel of this.subscriptions) {
      this.subscribe(channel);
    }
  }

  /**
   * WebSocketメッセージを処理する
   * @param event WebSocketのメッセージイベント
   */
  private handleMessage(event: MessageEvent): void {
    try {
      let data;
      
      // バイナリデータの場合はデコード
      if (event.data instanceof ArrayBuffer) {
        const decoder = new TextDecoder();
        data = JSON.parse(decoder.decode(event.data));
      } else {
        data = JSON.parse(event.data);
      }
      
      // メッセージタイプを取得
      const messageType = data.op || data.type || 'unknown';
      
      // メッセージタイプに対応するハンドラを呼び出す
      const handlers = this.messageHandlers.get(messageType) || [];
      handlers.forEach(handler => handler(data));
      
      // 'all'タイプのハンドラも呼び出す（すべてのメッセージを受け取る）
      const allHandlers = this.messageHandlers.get('all') || [];
      allHandlers.forEach(handler => handler(data));
    } catch (error) {
      console.error('Error handling WebSocket message:', error, event.data);
    }
  }

  /**
   * WebSocketメッセージハンドラを追加する
   * @param type メッセージタイプ
   * @param handler ハンドラ関数
   */
  addMessageHandler(type: string, handler: WebSocketMessageHandler): void {
    const handlers = this.messageHandlers.get(type) || [];
    handlers.push(handler);
    this.messageHandlers.set(type, handlers);
  }

  /**
   * WebSocketメッセージハンドラを削除する
   * @param type メッセージタイプ
   * @param handler ハンドラ関数
   */
  removeMessageHandler(type: string, handler: WebSocketMessageHandler): void {
    const handlers = this.messageHandlers.get(type) || [];
    const index = handlers.indexOf(handler);
    
    if (index !== -1) {
      handlers.splice(index, 1);
      this.messageHandlers.set(type, handlers);
    }
  }

  /**
   * WebSocketにメッセージを送信する
   * @param message 送信するメッセージ
   * @returns 送信が成功したかどうか
   */
  sendMessage(message: any): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.error('WebSocket is not open');
      return false;
    }
    
    try {
      this.ws.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('Error sending WebSocket message:', error);
      return false;
    }
  }

  /**
   * チャンネルをサブスクライブする
   * @param channel サブスクライブするチャンネル
   * @returns サブスクリプションが成功したかどうか
   */
  subscribe(channel: string): boolean {
    const message = {
      op: 'subscribe',
      channel: channel
    };
    
    const success = this.sendMessage(message);
    
    if (success) {
      this.subscriptions.add(channel);
    }
    
    return success;
  }

  /**
   * チャンネルをアンサブスクライブする
   * @param channel アンサブスクライブするチャンネル
   * @returns アンサブスクリプションが成功したかどうか
   */
  unsubscribe(channel: string): boolean {
    const message = {
      op: 'unsubscribe',
      channel: channel
    };
    
    const success = this.sendMessage(message);
    
    if (success) {
      this.subscriptions.delete(channel);
    }
    
    return success;
  }
}

/**
 * 環境変数から認証情報を取得してHyperLiquidWebSocketClientインスタンスを作成する
 * @param options WebSocketオプション
 * @returns HyperLiquidWebSocketClientインスタンス
 */
export function createWebSocketClientFromEnv(options?: Partial<WebSocketOptions>): HyperLiquidWebSocketClient {
  const auth = new HyperLiquidAuth();
  return new HyperLiquidWebSocketClient(auth, options);
}

// シングルトンインスタンス
let webSocketClientInstance: HyperLiquidWebSocketClient | null = null;

/**
 * HyperLiquidWebSocketClientのシングルトンインスタンスを取得する
 * @param options WebSocketオプション
 * @returns HyperLiquidWebSocketClientのシングルトンインスタンス
 */
export function getWebSocketClient(options?: Partial<WebSocketOptions>): HyperLiquidWebSocketClient {
  if (!webSocketClientInstance) {
    webSocketClientInstance = createWebSocketClientFromEnv(options);
  }
  return webSocketClientInstance;
}
