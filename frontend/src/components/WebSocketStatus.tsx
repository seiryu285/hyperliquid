import React, { useEffect, useState } from 'react';
import { Box, Chip, Tooltip } from '@mui/material';
import SignalWifiStatusbar4BarIcon from '@mui/icons-material/SignalWifiStatusbar4Bar';
import SignalWifiStatusbarConnectedNoInternet4Icon from '@mui/icons-material/SignalWifiStatusbarConnectedNoInternet4';
import SignalWifiStatusbarNullIcon from '@mui/icons-material/SignalWifiStatusbarNull';
import { WebSocketState } from '../api/webSocketClient';

interface WebSocketStatusProps {
  state: WebSocketState;
  latency?: number | null;
}

const WebSocketStatus: React.FC<WebSocketStatusProps> = ({ state, latency }) => {
  const [statusColor, setStatusColor] = useState<'success' | 'warning' | 'error' | 'default'>('default');
  const [statusText, setStatusText] = useState<string>('未接続');
  const [statusIcon, setStatusIcon] = useState<React.ReactNode>(<SignalWifiStatusbarNullIcon />);

  useEffect(() => {
    switch (state) {
      case WebSocketState.CONNECTING:
        setStatusColor('warning');
        setStatusText('接続中...');
        setStatusIcon(<SignalWifiStatusbarNullIcon />);
        break;
      case WebSocketState.OPEN:
        setStatusColor('warning');
        setStatusText('接続済み（未認証）');
        setStatusIcon(<SignalWifiStatusbarConnectedNoInternet4Icon />);
        break;
      case WebSocketState.AUTHENTICATED:
        setStatusColor('success');
        setStatusText('接続済み');
        setStatusIcon(<SignalWifiStatusbar4BarIcon />);
        break;
      case WebSocketState.CLOSED:
        setStatusColor('error');
        setStatusText('切断');
        setStatusIcon(<SignalWifiStatusbarNullIcon />);
        break;
      case WebSocketState.ERROR:
        setStatusColor('error');
        setStatusText('エラー');
        setStatusIcon(<SignalWifiStatusbarConnectedNoInternet4Icon />);
        break;
      default:
        setStatusColor('default');
        setStatusText('未接続');
        setStatusIcon(<SignalWifiStatusbarNullIcon />);
    }
  }, [state]);

  const tooltipText = latency 
    ? `WebSocket: ${statusText} (レイテンシ: ${latency}ms)` 
    : `WebSocket: ${statusText}`;

  return (
    <Tooltip title={tooltipText}>
      <Chip
        icon={<Box sx={{ display: 'flex', alignItems: 'center' }}>{statusIcon}</Box>}
        label={latency ? `${latency}ms` : statusText}
        size="small"
        color={statusColor}
        variant="outlined"
      />
    </Tooltip>
  );
};

export default WebSocketStatus;
