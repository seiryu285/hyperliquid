import { Subject, Observable } from 'rxjs';

export interface MarketData {
  timestamp: number;
  currentPrice: number;
  bidPrice: number;
  askPrice: number;
  volume24h: number;
  fundingRate: number;
  orderbook: {
    bids: [number, number][];
    asks: [number, number][];
  };
}

export interface PositionData {
  timestamp: number;
  size: number;
  entryPrice: number;
  currentMargin: number;
  requiredMargin: number;
  leverage: number;
  unrealizedPnl: number;
  liquidationPrice: number;
}

export interface RiskMetrics {
  marginBufferRatio: number;
  volatility: number;
  liquidationRisk: number;
  valueAtRisk: number;
  timestamp: number;
}

export type WebSocketMessage = {
  type: 'MARKET_DATA' | 'POSITION_DATA' | 'RISK_METRICS' | 'ERROR';
  payload: MarketData | PositionData | RiskMetrics | Error;
  timestamp: number;
};

export class WebSocketService {
  private ws: WebSocket | null = null;
  private readonly url: string;
  private reconnectAttempts = 0;
  private readonly maxReconnectAttempts = 5;
  private readonly reconnectDelay = 1000;
  private pingInterval: number | null = null;

  public marketData$ = new Subject<MarketData>();
  public positionData$ = new Subject<PositionData>();
  public riskMetrics$ = new Subject<RiskMetrics>();
  public connectionStatus$ = new Subject<'connected' | 'disconnected' | 'reconnecting'>();

  constructor(url: string = 'ws://localhost:8000/ws') {
    this.url = url;
  }

  public connect(): void {
    try {
      this.ws = new WebSocket(this.url);
      this.setupEventHandlers();
      this.startPingInterval();
    } catch (error) {
      console.error('WebSocket connection failed:', error);
      this.handleReconnection();
    }
  }

  private setupEventHandlers(): void {
    if (!this.ws) return;

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      this.connectionStatus$.next('connected');
      
      // Send initial subscription message
      this.sendMessage({
        type: 'SUBSCRIBE',
        channels: ['market_data', 'position_data', 'risk_metrics']
      });
    };

    this.ws.onmessage = (event: MessageEvent) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        this.handleIncomingMessage(message);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    this.ws.onclose = () => {
      console.log('WebSocket connection closed');
      this.connectionStatus$.next('disconnected');
      this.clearPingInterval();
      this.handleReconnection();
    };

    this.ws.onerror = (error: Event) => {
      console.error('WebSocket error:', error);
      this.connectionStatus$.next('disconnected');
    };
  }

  private handleIncomingMessage(message: WebSocketMessage): void {
    switch (message.type) {
      case 'MARKET_DATA':
        this.marketData$.next(message.payload as MarketData);
        break;
      case 'POSITION_DATA':
        this.positionData$.next(message.payload as PositionData);
        break;
      case 'RISK_METRICS':
        this.riskMetrics$.next(message.payload as RiskMetrics);
        break;
      case 'ERROR':
        console.error('Server error:', (message.payload as Error).message);
        break;
      default:
        console.warn('Unknown message type:', message);
    }
  }

  private handleReconnection(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      this.connectionStatus$.next('reconnecting');
      console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
      setTimeout(() => this.connect(), this.reconnectDelay * this.reconnectAttempts);
    } else {
      console.error('Max reconnection attempts reached');
      this.connectionStatus$.next('disconnected');
    }
  }

  private startPingInterval(): void {
    this.pingInterval = window.setInterval(() => {
      if (this.isConnected()) {
        this.sendMessage({ type: 'PING', timestamp: Date.now() });
      }
    }, 30000) as unknown as number;
  }

  private clearPingInterval(): void {
    if (this.pingInterval !== null) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  private sendMessage(message: any): void {
    if (this.ws && this.isConnected()) {
      this.ws.send(JSON.stringify(message));
    }
  }

  public disconnect(): void {
    this.clearPingInterval();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  public isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  public getObservable<T>(type: 'market' | 'position' | 'risk'): Observable<T> {
    switch (type) {
      case 'market':
        return this.marketData$ as unknown as Observable<T>;
      case 'position':
        return this.positionData$ as unknown as Observable<T>;
      case 'risk':
        return this.riskMetrics$ as unknown as Observable<T>;
      default:
        throw new Error(`Unknown observable type: ${type}`);
    }
  }
}
