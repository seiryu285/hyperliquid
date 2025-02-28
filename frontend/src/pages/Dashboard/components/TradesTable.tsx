import React from 'react';
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableContainer, 
  TableHead, 
  TableRow, 
  Paper,
  Typography
} from '@mui/material';
import { format } from 'date-fns';

interface Trade {
  id: string;
  price: number;
  size: number;
  side: 'buy' | 'sell';
  timestamp: string;
}

interface TradesTableProps {
  data: Trade[];
}

const TradesTable: React.FC<TradesTableProps> = ({ data }) => {
  // Sort trades by timestamp (newest first)
  const sortedTrades = [...data].sort((a, b) => {
    return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
  });
  
  return (
    <TableContainer component={Paper} sx={{ maxHeight: 400 }}>
      <Table stickyHeader size="small">
        <TableHead>
          <TableRow>
            <TableCell>時間</TableCell>
            <TableCell>価格</TableCell>
            <TableCell align="right">数量</TableCell>
            <TableCell align="right">合計</TableCell>
            <TableCell>タイプ</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {sortedTrades.map((trade) => (
            <TableRow key={trade.id}>
              <TableCell>
                {format(new Date(trade.timestamp), 'HH:mm:ss')}
              </TableCell>
              <TableCell sx={{ color: trade.side === 'buy' ? '#26a69a' : '#ef5350' }}>
                {trade.price.toFixed(2)}
              </TableCell>
              <TableCell align="right">{trade.size.toFixed(4)}</TableCell>
              <TableCell align="right">{(trade.price * trade.size).toFixed(2)}</TableCell>
              <TableCell>
                <Typography
                  variant="body2"
                  sx={{
                    display: 'inline-block',
                    px: 1,
                    py: 0.5,
                    borderRadius: 1,
                    backgroundColor: trade.side === 'buy' ? 'rgba(38, 166, 154, 0.1)' : 'rgba(239, 83, 80, 0.1)',
                    color: trade.side === 'buy' ? '#26a69a' : '#ef5350',
                  }}
                >
                  {trade.side === 'buy' ? '買い' : '売り'}
                </Typography>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

export default TradesTable;
