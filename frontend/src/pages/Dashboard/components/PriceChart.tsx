import React, { useEffect, useRef } from 'react';
import { Box, useTheme, Typography } from '@mui/material';
import { createChart, CrosshairMode, IChartApi, ISeriesApi } from 'lightweight-charts';

interface PriceChartProps {
  data?: any[] | null;
  timeframe?: string;
}

const PriceChart: React.FC<PriceChartProps> = ({ data, timeframe = '1h' }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const theme = useTheme();

  useEffect(() => {
    if (!chartContainerRef.current || !data || data.length === 0) {
      return;
    }

    // Dispose previous chart if it exists
    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
      candlestickSeriesRef.current = null;
      volumeSeriesRef.current = null;
    }

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: chartContainerRef.current.clientHeight,
      layout: {
        background: { type: 'solid', color: theme.palette.background.paper },
        textColor: theme.palette.text.primary,
      },
      grid: {
        vertLines: { color: theme.palette.divider },
        horzLines: { color: theme.palette.divider },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          width: 1,
          color: theme.palette.primary.main,
          style: 1,
        },
        horzLine: {
          width: 1,
          color: theme.palette.primary.main,
          style: 1,
        },
      },
      timeScale: {
        borderColor: theme.palette.divider,
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: theme.palette.divider,
      },
    });

    // Add price series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    });

    // Add volume series
    const volumeSeries = chart.addHistogramSeries({
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: '',
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    });

    // Format data for chart
    const formattedData = data.map(item => ({
      time: item.time as any,
      open: item.open,
      high: item.high,
      low: item.low,
      close: item.close,
    }));

    const volumeData = data.map(item => ({
      time: item.time as any,
      value: item.volume,
      color: item.close >= item.open ? '#26a69a' : '#ef5350',
    }));

    // Set data
    candlestickSeries.setData(formattedData);
    volumeSeries.setData(volumeData);

    // Fit content
    chart.timeScale().fitContent();

    // Store references
    chartRef.current = chart;
    candlestickSeriesRef.current = candlestickSeries;
    volumeSeriesRef.current = volumeSeries;

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [data, theme, timeframe]);

  if (!data || data.length === 0) {
    return (
      <Box
        sx={{
          width: '100%',
          height: '100%',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
        }}
      >
        <Typography>No data available</Typography>
      </Box>
    );
  }

  return (
    <Box
      ref={chartContainerRef}
      sx={{
        width: '100%',
        height: '100%',
        '& .tv-lightweight-charts': {
          width: '100% !important',
          height: '100% !important',
        },
      }}
    />
  );
};

export default PriceChart;
