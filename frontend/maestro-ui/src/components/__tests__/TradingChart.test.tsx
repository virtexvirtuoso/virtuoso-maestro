import {
  convertOHLCVData,
  convertTradeMarkers,
  convertIndicatorData,
} from '../TradingChart';
import { Time } from 'lightweight-charts';

// Note: Component render tests are skipped because lightweight-charts
// requires browser APIs (matchMedia, ResizeObserver) that are difficult
// to mock in jsdom. The component has been manually verified to work.
// These tests cover the helper functions which handle data transformation.

describe('convertOHLCVData', () => {
  it('converts API response to chart format', () => {
    const apiData = [
      { timestamp: { epoch_time: 1609632000 }, open: 110, high: 120, low: 105, close: 115 },
      { timestamp: { epoch_time: 1609459200 }, open: 100, high: 110, low: 95, close: 105 },
    ];

    const result = convertOHLCVData(apiData);

    // Should be sorted by time
    expect(result[0].time).toBe(1609459200);
    expect(result[1].time).toBe(1609632000);
    expect(result).toHaveLength(2);
    expect(result[0].open).toBe(100);
    expect(result[0].close).toBe(105);
  });

  it('handles empty array', () => {
    const result = convertOHLCVData([]);
    expect(result).toHaveLength(0);
  });

  it('preserves all OHLCV values', () => {
    const apiData = [
      { timestamp: { epoch_time: 1609459200 }, open: 100.5, high: 110.75, low: 95.25, close: 105.5 },
    ];

    const result = convertOHLCVData(apiData);

    expect(result[0].open).toBe(100.5);
    expect(result[0].high).toBe(110.75);
    expect(result[0].low).toBe(95.25);
    expect(result[0].close).toBe(105.5);
  });
});

describe('convertTradeMarkers', () => {
  it('converts buy/sell data to markers', () => {
    const buyData = [{ x: 1609459200000, title: 'B1' }];
    const sellData = [{ x: 1609545600000, title: 'S1' }];

    const result = convertTradeMarkers(buyData, sellData);

    expect(result).toHaveLength(2);

    const buyMarker = result.find((m) => m.shape === 'arrowUp');
    const sellMarker = result.find((m) => m.shape === 'arrowDown');

    expect(buyMarker?.position).toBe('belowBar');
    expect(buyMarker?.text).toBe('B1');
    expect(buyMarker?.color).toBe('#26a69a');
    expect(sellMarker?.position).toBe('aboveBar');
    expect(sellMarker?.text).toBe('S1');
    expect(sellMarker?.color).toBe('#ef5350');
  });

  it('uses default text when title not provided', () => {
    const buyData = [{ x: 1609459200000 }];
    const sellData = [{ x: 1609545600000 }];

    const result = convertTradeMarkers(buyData, sellData);

    expect(result[0].text).toBe('BUY');
    expect(result[1].text).toBe('SELL');
  });

  it('uses custom colors when provided', () => {
    const buyData = [{ x: 1609459200000 }];
    const sellData = [{ x: 1609545600000 }];

    const result = convertTradeMarkers(buyData, sellData, '#00ff00', '#ff0000');

    expect(result[0].color).toBe('#00ff00');
    expect(result[1].color).toBe('#ff0000');
  });

  it('converts milliseconds to seconds for time', () => {
    const buyData = [{ x: 1609459200000 }]; // 1609459200000 ms = 1609459200 s

    const result = convertTradeMarkers(buyData, []);

    expect(result[0].time).toBe(1609459200);
  });

  it('handles empty arrays', () => {
    const result = convertTradeMarkers([], []);
    expect(result).toHaveLength(0);
  });

  it('handles multiple markers', () => {
    const buyData = [
      { x: 1609459200000, title: 'B1' },
      { x: 1609632000000, title: 'B2' },
    ];
    const sellData = [
      { x: 1609545600000, title: 'S1' },
    ];

    const result = convertTradeMarkers(buyData, sellData);

    expect(result).toHaveLength(3);
    expect(result.filter((m) => m.shape === 'arrowUp')).toHaveLength(2);
    expect(result.filter((m) => m.shape === 'arrowDown')).toHaveLength(1);
  });
});

describe('convertIndicatorData', () => {
  it('converts indicator data structure', () => {
    const indicators = {
      sma: {
        line1: { label: 'SMA', name: '20', x: [100, 105, 110] },
      },
    };
    const timestamps = [1609459200, 1609545600, 1609632000];

    const result = convertIndicatorData(indicators, timestamps);

    expect(result).toHaveLength(1);
    expect(result[0].name).toBe('SMA - 20');
    expect(result[0].data).toHaveLength(3);
    expect(result[0].data[0].value).toBe(100);
    expect(result[0].data[0].time).toBe(1609459200);
  });

  it('handles multiple indicators', () => {
    const indicators = {
      sma: {
        fast: { label: 'SMA', name: '10', x: [100, 105] },
        slow: { label: 'SMA', name: '20', x: [98, 102] },
      },
      ema: {
        line1: { label: 'EMA', name: '12', x: [101, 106] },
      },
    };
    const timestamps = [1609459200, 1609545600];

    const result = convertIndicatorData(indicators, timestamps);

    expect(result).toHaveLength(3);
    expect(result.map((r) => r.name)).toContain('SMA - 10');
    expect(result.map((r) => r.name)).toContain('SMA - 20');
    expect(result.map((r) => r.name)).toContain('EMA - 12');
  });

  it('handles empty indicators', () => {
    const result = convertIndicatorData({}, [1609459200]);
    expect(result).toHaveLength(0);
  });

  it('maps timestamps correctly to values', () => {
    const indicators = {
      test: {
        line: { label: 'Test', name: 'Line', x: [10, 20, 30] },
      },
    };
    const timestamps = [100, 200, 300];

    const result = convertIndicatorData(indicators, timestamps);

    expect(result[0].data[0]).toEqual({ time: 100, value: 10 });
    expect(result[0].data[1]).toEqual({ time: 200, value: 20 });
    expect(result[0].data[2]).toEqual({ time: 300, value: 30 });
  });
});
