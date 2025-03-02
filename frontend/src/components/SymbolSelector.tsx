import React, { useState, useEffect } from 'react';
import {
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
  Chip,
  Typography,
  CircularProgress
} from '@mui/material';
import { getAvailableSymbols } from '../api/marketData';

interface SymbolSelectorProps {
  selectedSymbol: string;
  onSymbolChange: (symbol: string) => void;
}

interface SymbolInfo {
  symbol: string;
  name: string;
  type: 'spot' | 'perp' | 'futures';
  baseAsset: string;
  quoteAsset: string;
}

const SymbolSelector: React.FC<SymbolSelectorProps> = ({ selectedSymbol, onSymbolChange }) => {
  const [symbols, setSymbols] = useState<SymbolInfo[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchSymbols = async () => {
      setLoading(true);
      try {
        const availableSymbols = await getAvailableSymbols();
        setSymbols(availableSymbols);
        setError(null);
      } catch (err) {
        console.error('Failed to fetch symbols:', err);
        setError('銘柄情報の取得に失敗しました');
        // フォールバックとして基本的な銘柄を設定
        setSymbols([
          { symbol: 'ETH-PERP', name: 'Ethereum Perpetual', type: 'perp', baseAsset: 'ETH', quoteAsset: 'USD' },
          { symbol: 'BTC-PERP', name: 'Bitcoin Perpetual', type: 'perp', baseAsset: 'BTC', quoteAsset: 'USD' },
          { symbol: 'SOL-PERP', name: 'Solana Perpetual', type: 'perp', baseAsset: 'SOL', quoteAsset: 'USD' },
          { symbol: 'AVAX-PERP', name: 'Avalanche Perpetual', type: 'perp', baseAsset: 'AVAX', quoteAsset: 'USD' },
          { symbol: 'ARB-PERP', name: 'Arbitrum Perpetual', type: 'perp', baseAsset: 'ARB', quoteAsset: 'USD' }
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchSymbols();
  }, []);

  const handleChange = (event: SelectChangeEvent<string>) => {
    onSymbolChange(event.target.value);
  };

  const getSymbolTypeChip = (type: 'spot' | 'perp' | 'futures') => {
    let color: 'primary' | 'secondary' | 'default' = 'default';
    let label = type;

    switch (type) {
      case 'perp':
        color = 'primary';
        label = '無期限';
        break;
      case 'futures':
        color = 'secondary';
        label = '先物';
        break;
      case 'spot':
        color = 'default';
        label = '現物';
        break;
    }

    return <Chip size="small" color={color} label={label} sx={{ ml: 1 }} />;
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center' }}>
        <CircularProgress size={20} sx={{ mr: 1 }} />
        <Typography variant="body2">銘柄を読み込み中...</Typography>
      </Box>
    );
  }

  if (error && symbols.length === 0) {
    return (
      <Typography variant="body2" color="error">
        {error}
      </Typography>
    );
  }

  return (
    <FormControl fullWidth size="small">
      <InputLabel id="symbol-select-label">銘柄</InputLabel>
      <Select
        labelId="symbol-select-label"
        id="symbol-select"
        value={selectedSymbol}
        label="銘柄"
        onChange={handleChange}
        renderValue={(value) => {
          const symbol = symbols.find((s) => s.symbol === value);
          return (
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              {symbol?.baseAsset || value}
              {symbol && getSymbolTypeChip(symbol.type)}
            </Box>
          );
        }}
      >
        {symbols.map((symbol) => (
          <MenuItem key={symbol.symbol} value={symbol.symbol}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
              <Typography variant="body2">{symbol.name}</Typography>
              {getSymbolTypeChip(symbol.type)}
            </Box>
          </MenuItem>
        ))}
      </Select>
    </FormControl>
  );
};

export default SymbolSelector;
