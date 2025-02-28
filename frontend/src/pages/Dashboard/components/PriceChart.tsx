import React, { useEffect, useRef } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData } from 'lightweight-charts';
import { Box } from '@mui/material';

interface PriceChartProps {
  data: any[];
}

const PriceChart: React.FC<PriceChartProps> = ({ data }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  
  useEffect(() => {
    if (chartContainerRef.current) {
      // Create chart
      const chart = createChart(chartContainerRef.current, {
        width: chartContainerRef.current.clientWidth,
        height: chartContainerRef.current.clientHeight,
        layout: {
          background: { color: '#ffffff' },
          textColor: '#333',
        },
        grid: {
          vertLines: { color: '#f0f0f0' },
          horzLines: { color: '#f0f0f0' },
        },
        timeScale: {
          timeVisible: true,
          secondsVisible: false,
          borderColor: '#D1D4DC',
        },
        rightPriceScale: {
          borderColor: '#D1D4DC',
        },
      });
      
      // Create candlestick series
      const candlestickSeries = chart.addCandlestickSeries({
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderVisible: false,
        wickUpColor: '#26a69a',
        wickDownColor: '#ef5350',
      });
      
      // Create volume series
      const volumeSeries = chart.addHistogramSeries({
        color: '#26a69a',
        priceFormat: {
          type: 'volume',
        },
        priceScaleId: '',
        scaleMargins: {
          top: 0.8,
          bottom: 0,
        },
      });
      
      // Save references
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
      
      // Clean up
      return () => {
        window.removeEventListener('resize', handleResize);
        if (chartRef.current) {
          chartRef.current.remove();
          chartRef.current = null;
          candlestickSeriesRef.current = null;
          volumeSeriesRef.current = null;
        }
      };
    }
  }, []);
  
  useEffect(() => {
    if (candlestickSeriesRef.current && volumeSeriesRef.current && data.length > 0) {
      // Format data for candlestick chart
      const candlestickData = data.map(item => ({
        time: new Date(item.timestamp).getTime() / 1000,
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close,
      }));
      
      // Format data for volume chart
      const volumeData = data.map(item => ({
        time: new Date(item.timestamp).getTime() / 1000,
        value: item.volume,
        color: item.close >= item.open ? '#26a69a' : '#ef5350',
      }));
      
      // Set data
      candlestickSeriesRef.current.setData(candlestickData as CandlestickData[]);
      volumeSeriesRef.current.setData(volumeData);
      
      // Fit content
      if (chartRef.current) {
        chartRef.current.timeScale().fitContent();
      }
    }
  }, [data]);
  
  return (
    <Box ref={chartContainerRef} sx={{ width: '100%', height: '100%' }} />
  );
};

export default PriceChart;
