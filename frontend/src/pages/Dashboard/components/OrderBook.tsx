import React from 'react';
import { Box, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Typography } from '@mui/material';

interface OrderBookProps {
  data: {
    bids: Array<[number, number]>; // [price, size]
    asks: Array<[number, number]>; // [price, size]
  };
}

const OrderBook: React.FC<OrderBookProps> = ({ data }) => {
  const { bids, asks } = data;
  
  // Sort bids in descending order (highest price first)
  const sortedBids = [...bids].sort((a, b) => b[0] - a[0]);
  
  // Sort asks in ascending order (lowest price first)
  const sortedAsks = [...asks].sort((a, b) => a[0] - b[0]);
  
  // Take only top 10 entries for each side
  const topBids = sortedBids.slice(0, 10);
  const topAsks = sortedAsks.slice(0, 10);
  
  // Calculate total size for depth visualization
  const maxBidSize = Math.max(...topBids.map(bid => bid[1]));
  const maxAskSize = Math.max(...topAsks.map(ask => ask[1]));
  const maxSize = Math.max(maxBidSize, maxAskSize);
  
  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Asks (Sell orders) */}
      <TableContainer sx={{ flex: 1, overflow: 'auto' }}>
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell>価格</TableCell>
              <TableCell align="right">数量</TableCell>
              <TableCell align="right">合計</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {topAsks.map(([price, size], index) => {
              const depthPercentage = (size / maxSize) * 100;
              
              return (
                <TableRow key={`ask-${index}`} sx={{ position: 'relative' }}>
                  <TableCell sx={{ color: '#ef5350' }}>
                    {price.toFixed(2)}
                  </TableCell>
                  <TableCell align="right">{size.toFixed(4)}</TableCell>
                  <TableCell align="right">{(price * size).toFixed(2)}</TableCell>
                  <Box
                    sx={{
                      position: 'absolute',
                      right: 0,
                      top: 0,
                      bottom: 0,
                      width: `${depthPercentage}%`,
                      backgroundColor: 'rgba(239, 83, 80, 0.15)',
                      zIndex: -1,
                    }}
                  />
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
      
      {/* Spread */}
      <Box sx={{ py: 1, textAlign: 'center', borderTop: '1px solid #eee', borderBottom: '1px solid #eee' }}>
        <Typography variant="body2">
          スプレッド: {topAsks.length > 0 && topBids.length > 0
            ? (topAsks[0][0] - topBids[0][0]).toFixed(2)
            : '-'
          } ({topAsks.length > 0 && topBids.length > 0
            ? ((topAsks[0][0] - topBids[0][0]) / topBids[0][0] * 100).toFixed(3)
            : '-'
          }%)
        </Typography>
      </Box>
      
      {/* Bids (Buy orders) */}
      <TableContainer sx={{ flex: 1, overflow: 'auto' }}>
        <Table size="small">
          <TableBody>
            {topBids.map(([price, size], index) => {
              const depthPercentage = (size / maxSize) * 100;
              
              return (
                <TableRow key={`bid-${index}`} sx={{ position: 'relative' }}>
                  <TableCell sx={{ color: '#26a69a' }}>
                    {price.toFixed(2)}
                  </TableCell>
                  <TableCell align="right">{size.toFixed(4)}</TableCell>
                  <TableCell align="right">{(price * size).toFixed(2)}</TableCell>
                  <Box
                    sx={{
                      position: 'absolute',
                      right: 0,
                      top: 0,
                      bottom: 0,
                      width: `${depthPercentage}%`,
                      backgroundColor: 'rgba(38, 166, 154, 0.15)',
                      zIndex: -1,
                    }}
                  />
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
};

export default OrderBook;
