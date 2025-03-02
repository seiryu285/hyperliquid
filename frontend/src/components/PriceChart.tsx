import React, { useEffect, useRef } from 'react';
import { Box, useTheme } from '@mui/material';
import * as echarts from 'echarts';
import { OHLCVData } from '../types/market';

// @ts-ignore
import { graphic } from 'echarts';

interface PriceChartProps {
  data: OHLCVData[];
  theme?: 'light' | 'dark';
  height?: number;
  width?: string;
  is3D?: boolean;
}

const PriceChart: React.FC<PriceChartProps> = ({ 
  data, 
  theme = 'dark', 
  height = 400, 
  width = '100%',
  is3D = false
}) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    // チャートの初期化
    if (chartRef.current) {
      if (!chartInstance.current) {
        chartInstance.current = echarts.init(chartRef.current);
      }
      
      // データがない場合は何もしない
      if (data.length === 0) {
        return;
      }

      // データの準備
      const dates = data.map(item => new Date(item.time).toLocaleString());
      const values = data.map(item => [item.open, item.close, item.low, item.high]);
      const volumes = data.map(item => item.volume);
      
      // 上昇・下降の判定
      const upColor = '#00da3c';
      const downColor = '#ec0000';
      
      const volumeColors = data.map(item => item.close > item.open ? upColor : downColor);
      
      // オプションの設定
      const option: echarts.EChartsOption = {
        animation: true,
        legend: {
          bottom: 10,
          left: 'center',
          data: ['OHLC', 'MA5', 'MA10', 'MA20', 'MA30']
        },
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            type: 'cross'
          },
          borderWidth: 1,
          borderColor: '#ccc',
          padding: 10,
          textStyle: {
            color: '#000'
          },
          position: function (pos, params, el, elRect, size) {
            const obj: Record<string, number> = {};
            obj[['left', 'right'][+(pos[0] < size.viewSize[0] / 2)]] = 30;
            obj[['top', 'bottom'][+(pos[1] < size.viewSize[1] / 2)]] = 30;
            return obj;
          }
        },
        axisPointer: {
          link: [{ xAxisIndex: 'all' }],
          label: {
            backgroundColor: '#777'
          }
        },
        toolbox: {
          feature: {
            dataZoom: {
              yAxisIndex: false
            },
            brush: {
              type: ['lineX', 'clear']
            }
          }
        },
        brush: {
          xAxisIndex: 'all',
          brushLink: 'all',
          outOfBrush: {
            colorAlpha: 0.1
          }
        },
        visualMap: {
          show: false,
          seriesIndex: 5,
          dimension: 2,
          pieces: [
            {
              value: 1,
              color: downColor
            },
            {
              value: -1,
              color: upColor
            }
          ]
        },
        grid: [
          {
            left: '10%',
            right: '8%',
            height: '50%'
          },
          {
            left: '10%',
            right: '8%',
            top: '63%',
            height: '16%'
          }
        ],
        xAxis: [
          {
            type: 'category',
            data: dates,
            boundaryGap: false,
            axisLine: { onZero: false },
            splitLine: { show: false },
            min: 'dataMin',
            max: 'dataMax',
            axisPointer: {
              z: 100
            }
          },
          {
            type: 'category',
            gridIndex: 1,
            data: dates,
            boundaryGap: false,
            axisLine: { onZero: false },
            axisTick: { show: false },
            splitLine: { show: false },
            axisLabel: { show: false },
            min: 'dataMin',
            max: 'dataMax'
          }
        ],
        yAxis: [
          {
            scale: true,
            splitArea: {
              show: true
            }
          },
          {
            scale: true,
            gridIndex: 1,
            splitNumber: 2,
            axisLabel: { show: false },
            axisLine: { show: false },
            axisTick: { show: false },
            splitLine: { show: false }
          }
        ],
        dataZoom: [
          {
            type: 'inside',
            xAxisIndex: [0, 1],
            start: 50,
            end: 100
          },
          {
            show: true,
            xAxisIndex: [0, 1],
            type: 'slider',
            top: '85%',
            start: 50,
            end: 100
          }
        ],
        series: [
          {
            name: 'OHLC',
            type: 'candlestick',
            data: values,
            itemStyle: {
              color: upColor,
              color0: downColor,
              borderColor: undefined,
              borderColor0: undefined
            },
            markPoint: {
              label: {
                formatter: function (param: any) {
                  return param != null ? Math.round(param.value) + '' : '';
                }
              },
              data: [
                {
                  name: '最高値',
                  type: 'max',
                  valueDim: 'highest'
                },
                {
                  name: '最安値',
                  type: 'min',
                  valueDim: 'lowest'
                }
              ],
              tooltip: {
                formatter: function (param: any) {
                  return param.name + '<br>' + (param.data.coord || '');
                }
              }
            },
            markLine: {
              symbol: ['none', 'none'],
              data: [
                [
                  {
                    name: '最安値 → 最高値',
                    type: 'min',
                    valueDim: 'lowest',
                    symbol: 'circle',
                    symbolSize: 10,
                    lineStyle: { width: 1, color: '#555' },
                    label: { show: false }
                  },
                  {
                    type: 'max',
                    valueDim: 'highest',
                    symbol: 'circle',
                    symbolSize: 10,
                    lineStyle: { width: 1, color: '#555' },
                    label: { show: false }
                  }
                ]
              ]
            }
          },
          {
            name: 'Volume',
            type: 'bar',
            xAxisIndex: 1,
            yAxisIndex: 1,
            data: volumes,
            itemStyle: {
              color: function(params: any) {
                return volumeColors[params.dataIndex];
              }
            }
          }
        ]
      };

      // 3Dビューの設定
      if (is3D) {
        option.grid = [
          {
            left: '10%',
            right: '8%',
            height: '50%'
          },
          {
            left: '10%',
            right: '8%',
            top: '63%',
            height: '16%'
          }
        ];
        
        // 3D効果を追加
        if (option.series && Array.isArray(option.series) && option.series.length > 0) {
          const candlestickSeries = option.series[0];
          (candlestickSeries as any).type = 'custom';
          (candlestickSeries as any).renderItem = function (params: any, api: any) {
            const xValue = api.value(0);
            const openPoint = api.coord([xValue, api.value(1)]);
            const closePoint = api.coord([xValue, api.value(2)]);
            const lowPoint = api.coord([xValue, api.value(3)]);
            const highPoint = api.coord([xValue, api.value(4)]);
            const rectWidth = api.size([1, 0])[0] * 0.6;
            
            const style = api.style({
              stroke: api.visual('color'),
              fill: api.visual('color')
            });
            
            return {
              type: 'group',
              children: [
                {
                  type: 'rect',
                  shape: {
                    x: openPoint[0] - rectWidth / 2,
                    y: Math.min(openPoint[1], closePoint[1]),
                    width: rectWidth,
                    height: Math.abs(closePoint[1] - openPoint[1])
                  },
                  style
                },
                {
                  type: 'line',
                  shape: {
                    x1: lowPoint[0],
                    y1: lowPoint[1],
                    x2: highPoint[0],
                    y2: highPoint[1]
                  },
                  style
                }
              ]
            };
          };
        }
      }

      // チャートの描画
      chartInstance.current.setOption(option);
    }

    // リサイズハンドラ
    const handleResize = () => {
      chartInstance.current?.resize();
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [data, is3D]);

  // コンポーネントのアンマウント時にチャートを破棄
  useEffect(() => {
    return () => {
      if (chartInstance.current) {
        chartInstance.current.dispose();
        chartInstance.current = null;
      }
    };
  }, []);

  return (
    <Box sx={{ height: '100%', width: '100%', position: 'relative' }}>
      {data.length === 0 ? (
        <Box 
          sx={{ 
            height: '100%', 
            width: '100%', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center' 
          }}
        >
          <div>No data available</div>
        </Box>
      ) : (
        <div 
          ref={chartRef} 
          style={{ 
            height: '100%', 
            width: '100%' 
          }} 
        />
      )}
    </Box>
  );
};

export default PriceChart;
