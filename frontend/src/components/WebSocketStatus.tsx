import React, { useEffect, useState } from 'react';
import { Box, Chip, Tooltip, Badge, useTheme, alpha } from '@mui/material';
import SignalWifiStatusbar4BarIcon from '@mui/icons-material/SignalWifiStatusbar4Bar';
import SignalWifiStatusbarConnectedNoInternet4Icon from '@mui/icons-material/SignalWifiStatusbarConnectedNoInternet4';
import SignalWifiStatusbarNullIcon from '@mui/icons-material/SignalWifiStatusbarNull';
import { WebSocketState } from '../api/webSocketClient';

interface WebSocketStatusProps {
  state: WebSocketState;
  latency?: number | null;
}

const WebSocketStatus: React.FC<WebSocketStatusProps> = ({ state, latency }) => {
  const theme = useTheme();
  const [statusColor, setStatusColor] = useState<'success' | 'warning' | 'error' | 'default'>('default');
  const [statusText, setStatusText] = useState<string>('未接続');
  const [statusIcon, setStatusIcon] = useState<React.ReactNode>(<SignalWifiStatusbarNullIcon />);
  const [pulsing, setPulsing] = useState<boolean>(false);

  useEffect(() => {
    switch (state) {
      case WebSocketState.CONNECTING:
        setStatusColor('warning');
        setStatusText('接続中...');
        setStatusIcon(<SignalWifiStatusbarNullIcon />);
        setPulsing(true);
        break;
      case WebSocketState.OPEN:
        setStatusColor('warning');
        setStatusText('接続済み（未認証）');
        setStatusIcon(<SignalWifiStatusbarConnectedNoInternet4Icon />);
        setPulsing(false);
        break;
      case WebSocketState.AUTHENTICATED:
        setStatusColor('success');
        setStatusText('接続済み');
        setStatusIcon(<SignalWifiStatusbar4BarIcon />);
        setPulsing(false);
        break;
      case WebSocketState.CLOSED:
        setStatusColor('error');
        setStatusText('切断');
        setStatusIcon(<SignalWifiStatusbarNullIcon />);
        setPulsing(false);
        break;
      case WebSocketState.ERROR:
        setStatusColor('error');
        setStatusText('エラー');
        setStatusIcon(<SignalWifiStatusbarConnectedNoInternet4Icon />);
        setPulsing(true);
        break;
      default:
        setStatusColor('default');
        setStatusText('未接続');
        setStatusIcon(<SignalWifiStatusbarNullIcon />);
        setPulsing(false);
    }
  }, [state]);

  // レイテンシに基づいて接続品質を評価
  const getLatencyQuality = (): 'excellent' | 'good' | 'fair' | 'poor' => {
    if (!latency) return 'fair';
    if (latency < 50) return 'excellent';
    if (latency < 100) return 'good';
    if (latency < 200) return 'fair';
    return 'poor';
  };

  const latencyQuality = getLatencyQuality();
  const latencyColor = {
    excellent: theme.palette.success.main,
    good: theme.palette.success.light,
    fair: theme.palette.warning.main,
    poor: theme.palette.error.main
  }[latencyQuality];

  const tooltipText = latency 
    ? `WebSocket: ${statusText} (レイテンシ: ${latency}ms - ${
        latencyQuality === 'excellent' ? '優秀' : 
        latencyQuality === 'good' ? '良好' : 
        latencyQuality === 'fair' ? '普通' : '低速'
      })` 
    : `WebSocket: ${statusText}`;

  return (
    <Tooltip title={tooltipText} arrow>
      <Chip
        icon={
          <Box 
            sx={{ 
              display: 'flex', 
              alignItems: 'center',
              animation: pulsing ? `pulse 1.5s infinite ease-in-out` : 'none',
              '@keyframes pulse': {
                '0%': {
                  opacity: 0.6,
                },
                '50%': {
                  opacity: 1,
                },
                '100%': {
                  opacity: 0.6,
                }
              }
            }}
          >
            {latency ? (
              <Badge
                variant="dot"
                overlap="circular"
                sx={{
                  '& .MuiBadge-badge': {
                    backgroundColor: latencyColor,
                    boxShadow: `0 0 0 2px ${theme.palette.background.paper}`
                  }
                }}
              >
                {statusIcon}
              </Badge>
            ) : (
              statusIcon
            )}
          </Box>
        }
        label={latency ? `${latency}ms` : statusText}
        size="small"
        color={statusColor}
        sx={{
          fontWeight: 500,
          border: `1px solid ${alpha(theme.palette[statusColor].main, 0.5)}`,
          backgroundColor: alpha(theme.palette[statusColor].main, 0.1),
          '& .MuiChip-label': {
            px: 1,
          }
        }}
      />
    </Tooltip>
  );
};

export default WebSocketStatus;
