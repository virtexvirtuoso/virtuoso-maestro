// TradingChart.tsx - TradingView Lightweight Charts replacement for Highcharts
// Provides candlestick charts, trade markers, and indicator overlays

import React, { useEffect, useRef, useCallback } from 'react';
import {
  createChart,
  IChartApi,
  ISeriesApi,
  CandlestickData,
  LineData,
  SeriesMarker,
  Time,
  ColorType,
  LineWidth,
  CrosshairMode,
  LineStyle,
} from 'lightweight-charts';

// Types for chart data
export interface OHLCVData {
  time: Time;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface TradeMarker {
  time: Time;
  position: 'aboveBar' | 'belowBar';
  color: string;
  shape: 'arrowUp' | 'arrowDown' | 'circle' | 'square';
  text?: string;
  size?: number;
}

export interface IndicatorLine {
  name: string;
  data: LineData[];
  color?: string;
  lineWidth?: number;
}

export interface TradingChartProps {
  // OHLCV candlestick data
  ohlcvData: OHLCVData[];
  // Trade markers (buy/sell signals)
  markers?: TradeMarker[];
  // Indicator overlay lines
  indicators?: IndicatorLine[];
  // Chart dimensions
  width?: number;
  height?: number;
  // Custom styling
  upColor?: string;
  downColor?: string;
  // Chart title
  title?: string;
}

// Dark theme configuration matching the app's MUI dark theme with gold accent
const DARK_THEME = {
  layout: {
    background: { type: ColorType.Solid, color: '#121212' },
    textColor: '#e0e0e0',
  },
  grid: {
    vertLines: { color: 'rgba(255, 255, 255, 0.05)' },
    horzLines: { color: 'rgba(255, 255, 255, 0.05)' },
  },
  crosshair: {
    mode: CrosshairMode.Normal,
    vertLine: {
      width: 1 as LineWidth,
      color: '#fbbf24',
      style: LineStyle.Dashed,
      labelBackgroundColor: '#1e1e1e',
    },
    horzLine: {
      width: 1 as LineWidth,
      color: '#fbbf24',
      style: LineStyle.Dashed,
      labelBackgroundColor: '#1e1e1e',
    },
  },
  timeScale: {
    borderColor: 'rgba(255, 255, 255, 0.1)',
    timeVisible: true,
    secondsVisible: false,
  },
  rightPriceScale: {
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
};

export default function TradingChart({
  ohlcvData,
  markers = [],
  indicators = [],
  width,
  height = 400,
  upColor = '#26a69a',
  downColor = '#ef5350',
  title,
}: TradingChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const indicatorSeriesRef = useRef<ISeriesApi<'Line'>[]>([]);

  // Handle window resize
  const handleResize = useCallback(() => {
    if (chartRef.current && chartContainerRef.current) {
      const containerWidth = width || chartContainerRef.current.clientWidth;
      chartRef.current.applyOptions({
        width: containerWidth,
        height: height,
      });
    }
  }, [width, height]);

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart with dark theme
    const chart = createChart(chartContainerRef.current, {
      width: width || chartContainerRef.current.clientWidth,
      height: height,
      ...DARK_THEME,
    });
    chartRef.current = chart;

    // Add candlestick series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: upColor,
      downColor: downColor,
      borderUpColor: upColor,
      borderDownColor: downColor,
      wickUpColor: upColor,
      wickDownColor: downColor,
    });
    candlestickSeriesRef.current = candlestickSeries;

    // Add resize listener
    window.addEventListener('resize', handleResize);

    // Cleanup on unmount
    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
      candlestickSeriesRef.current = null;
      indicatorSeriesRef.current = [];
    };
  }, [width, height, upColor, downColor, handleResize]);

  // Update candlestick data
  useEffect(() => {
    if (!candlestickSeriesRef.current || !ohlcvData.length) return;

    // Sort data by time and set
    const sortedData = [...ohlcvData].sort((a, b) => {
      const timeA = typeof a.time === 'number' ? a.time : new Date(a.time as string).getTime();
      const timeB = typeof b.time === 'number' ? b.time : new Date(b.time as string).getTime();
      return timeA - timeB;
    });

    candlestickSeriesRef.current.setData(sortedData as CandlestickData[]);

    // Fit content to view
    if (chartRef.current) {
      chartRef.current.timeScale().fitContent();
    }
  }, [ohlcvData]);

  // Update markers (buy/sell signals)
  useEffect(() => {
    if (!candlestickSeriesRef.current) return;

    if (markers.length > 0) {
      // Convert to lightweight-charts marker format
      const chartMarkers: SeriesMarker<Time>[] = markers.map((m) => ({
        time: m.time,
        position: m.position,
        color: m.color,
        shape: m.shape,
        text: m.text || '',
        size: m.size || 1,
      }));

      // Sort markers by time
      chartMarkers.sort((a, b) => {
        const timeA = typeof a.time === 'number' ? a.time : new Date(a.time as string).getTime();
        const timeB = typeof b.time === 'number' ? b.time : new Date(b.time as string).getTime();
        return timeA - timeB;
      });

      candlestickSeriesRef.current.setMarkers(chartMarkers);
    } else {
      candlestickSeriesRef.current.setMarkers([]);
    }
  }, [markers]);

  // Update indicator overlays
  useEffect(() => {
    if (!chartRef.current) return;

    // Remove existing indicator series
    indicatorSeriesRef.current.forEach((series) => {
      if (chartRef.current) {
        chartRef.current.removeSeries(series);
      }
    });
    indicatorSeriesRef.current = [];

    // Add new indicator series
    const defaultColors = ['#2196f3', '#ff9800', '#9c27b0', '#4caf50', '#f44336'];
    indicators.forEach((indicator, index) => {
      if (!chartRef.current) return;

      const lineSeries = chartRef.current.addLineSeries({
        color: indicator.color || defaultColors[index % defaultColors.length],
        lineWidth: (indicator.lineWidth || 2) as LineWidth,
        title: indicator.name,
      });

      // Sort indicator data by time
      const sortedData = [...indicator.data].sort((a, b) => {
        const timeA = typeof a.time === 'number' ? a.time : new Date(a.time as string).getTime();
        const timeB = typeof b.time === 'number' ? b.time : new Date(b.time as string).getTime();
        return timeA - timeB;
      });

      lineSeries.setData(sortedData);
      indicatorSeriesRef.current.push(lineSeries);
    });
  }, [indicators]);

  return (
    <div style={{ position: 'relative' }}>
      {title && (
        <div
          style={{
            color: '#d1d4dc',
            fontSize: '14px',
            fontWeight: 500,
            marginBottom: '8px',
          }}
        >
          {title}
        </div>
      )}
      <div ref={chartContainerRef} style={{ width: '100%' }} />
    </div>
  );
}

// Helper function to convert API response to chart format
export function convertOHLCVData(
  data: Array<{
    timestamp: { epoch_time: number };
    open: number;
    high: number;
    low: number;
    close: number;
  }>
): OHLCVData[] {
  return data
    .sort((a, b) => a.timestamp.epoch_time - b.timestamp.epoch_time)
    .map((x) => ({
      time: x.timestamp.epoch_time as Time,
      open: x.open,
      high: x.high,
      low: x.low,
      close: x.close,
    }));
}

// Helper function to convert buy/sell data to markers
export function convertTradeMarkers(
  buyData: Array<{ x: number; title?: string }>,
  sellData: Array<{ x: number; title?: string }>,
  buyColor = '#26a69a',
  sellColor = '#ef5350'
): TradeMarker[] {
  const buyMarkers: TradeMarker[] = buyData.map((b) => ({
    time: (b.x / 1000) as Time, // Convert ms to seconds
    position: 'belowBar',
    color: buyColor,
    shape: 'arrowUp',
    text: b.title || 'BUY',
    size: 2,
  }));

  const sellMarkers: TradeMarker[] = sellData.map((s) => ({
    time: (s.x / 1000) as Time, // Convert ms to seconds
    position: 'aboveBar',
    color: sellColor,
    shape: 'arrowDown',
    text: s.title || 'SELL',
    size: 2,
  }));

  return [...buyMarkers, ...sellMarkers];
}

// Helper function to convert indicator data
export function convertIndicatorData(
  indicators: Record<string, Record<string, { label: string; name: string; x: number[] }>>,
  timestamps: number[]
): IndicatorLine[] {
  const result: IndicatorLine[] = [];

  for (const indicator in indicators) {
    for (const line in indicators[indicator]) {
      const lineData = indicators[indicator][line];
      result.push({
        name: `${lineData.label} - ${lineData.name}`,
        data: timestamps.map((ts, i) => ({
          time: ts as Time,
          value: lineData.x[i],
        })),
      });
    }
  }

  return result;
}
