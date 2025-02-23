import React, { useEffect, useRef, useState } from 'react';
import {
  Chart,
  ChartData,
  ChartOptions,
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  Title,
  CategoryScale,
  Tooltip
} from 'chart.js';
import { WebSocketService, RiskMetrics } from '../services/websocket';
import { styled } from '@mui/material/styles';

// Chart.jsのプラグイン登録
Chart.register(LineController, LineElement, PointElement, LinearScale, Title, CategoryScale, Tooltip);

const DashboardContainer = styled('div')`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  padding: 20px;
  background-color: #f5f5f5;
`;

const MetricCard = styled('div')`
  background: white;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
`;

const ChartContainer = styled('div')`
  grid-column: 1 / -1;
  background: white;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
`;

interface RiskDashboardProps {
  websocketService: WebSocketService;
}

export const RiskDashboard: React.FC<RiskDashboardProps> = ({ websocketService }) => {
  const [currentMetrics, setCurrentMetrics] = useState<RiskMetrics | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'reconnecting'>('disconnected');
  const chartRef = useRef<Chart<'line', number[], string> | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    // WebSocket購読
    const subscription = websocketService.riskMetrics$.subscribe({
      next: (metrics) => {
        setCurrentMetrics(metrics);
        updateChart(metrics);
        setConnectionStatus('connected');
      },
      error: () => {
        setConnectionStatus('disconnected');
      }
    });

    // Chart.jsの初期化
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        const chartData: ChartData<'line', number[], string> = {
          labels: [],
          datasets: [
            {
              label: 'Margin Buffer Ratio',
              data: [],
              borderColor: 'rgb(75, 192, 192)',
              tension: 0.1,
              fill: false
            },
            {
              label: 'Liquidation Risk',
              data: [],
              borderColor: 'rgb(255, 99, 132)',
              tension: 0.1,
              fill: false
            }
          ]
        };

        const chartOptions: ChartOptions<'line'> = {
          responsive: true,
          plugins: {
            title: {
              display: true,
              text: 'Risk Metrics Over Time'
            }
          },
          scales: {
            y: {
              beginAtZero: true,
              ticks: {
                callback: (value) => `${Number(value).toFixed(4)}`
              }
            }
          }
        };

        chartRef.current = new Chart(ctx, {
          type: 'line',
          data: chartData,
          options: chartOptions
        });
      }
    }

    // cleanup
    return () => {
      subscription.unsubscribe();
      if (chartRef.current) {
        chartRef.current.destroy();
      }
    };
  }, [websocketService]);

  // リアルタイムにチャートを更新
  const updateChart = (metrics: RiskMetrics) => {
    if (chartRef.current) {
      const chart = chartRef.current;
      const timestamp = new Date(metrics.timestamp).toLocaleTimeString();

      if (chart.data.labels && chart.data.datasets) {
        chart.data.labels.push(timestamp);
        chart.data.datasets[0].data.push(metrics.marginBufferRatio);
        chart.data.datasets[1].data.push(metrics.liquidationRisk);

        // 最新50件を保持
        if (chart.data.labels.length > 50) {
          chart.data.labels = chart.data.labels.slice(-50);
          chart.data.datasets.forEach(dataset => {
            dataset.data = dataset.data.slice(-50);
          });
        }

        chart.update('none'); // アニメーションなしで更新
      }
    }
  };

  // 指定した閾値を超えると赤、それ以外は緑にする
  const getMetricColor = (value: number, threshold: number): string => {
    return value > threshold ? 'red' : 'green';
  };

  return (
    <DashboardContainer>
      <div data-testid="connection-status" className={connectionStatus}>
        Connection Status: {connectionStatus}
      </div>

      <MetricCard data-testid="margin-buffer">
        <h3>Margin Buffer Ratio</h3>
        <p
          className="value"
          data-testid="margin-buffer-value"
          style={{ color: getMetricColor(currentMetrics?.marginBufferRatio ?? 0, 0.2) }}
        >
          {(currentMetrics?.marginBufferRatio ?? 0).toFixed(4)}
        </p>
      </MetricCard>

      <MetricCard data-testid="volatility">
        <h3>Volatility</h3>
        <p
          className="value"
          data-testid="volatility-value"
          style={{ color: getMetricColor(currentMetrics?.volatility ?? 0, 0.1) }}
        >
          {(currentMetrics?.volatility ?? 0).toFixed(4)}
        </p>
      </MetricCard>

      <MetricCard data-testid="liquidation-risk">
        <h3>Liquidation Risk</h3>
        <p
          className="value"
          data-testid="liquidation-risk-value"
          style={{ color: getMetricColor(currentMetrics?.liquidationRisk ?? 0, 0.05) }}
        >
          {(currentMetrics?.liquidationRisk ?? 0).toFixed(4)}
        </p>
      </MetricCard>

      <MetricCard data-testid="value-at-risk">
        <h3>Value at Risk</h3>
        <p
          className="value"
          data-testid="value-at-risk-value"
          style={{ color: getMetricColor(currentMetrics?.valueAtRisk ?? 0, 0.15) }}
        >
          {(currentMetrics?.valueAtRisk ?? 0).toFixed(4)}
        </p>
      </MetricCard>

      <ChartContainer>
        <canvas ref={canvasRef} data-testid="risk-chart" />
      </ChartContainer>
    </DashboardContainer>
  );
};
