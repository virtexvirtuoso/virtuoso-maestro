import React, { useEffect, useState } from 'react';
import { useTheme } from '@mui/material/styles';
import Highcharts from 'highcharts/highstock';
import HighchartsReact from 'highcharts-react-official';
import TradingChart, { convertOHLCVData, convertTradeMarkers, convertIndicatorData } from './TradingChart';
import { FEATURES } from '../config/featureFlags';

export default function StrategyChart({ data, provider, symbol, startDate, endDate }) {
  const theme = useTheme();
  const [chart, setChart] = useState(null);
  const [tradingViewData, setTradingViewData] = useState({
    ohlcvData: [],
    markers: [],
    indicators: [],
  });

  useEffect(() => {
    if (!data || !provider || !symbol) return;

    const indicators = data['indicators'];
    fetch(
      `${process.env.REACT_APP_REST_API_URL}/datasource/${provider}/${symbol}/1d/${startDate}/${endDate}`
    )
      .then((res) => res.json())
      .then((responseData) => {
        const sortedData = responseData['data'].sort(
          (a, b) => a['timestamp']['epoch_time'] - b['timestamp']['epoch_time']
        );

        if (FEATURES.USE_TRADINGVIEW) {
          // Convert data for TradingView Lightweight Charts
          const ohlcvData = convertOHLCVData(sortedData);

          // Convert buy/sell markers
          const markers = convertTradeMarkers(
            data['buy'] || [],
            data['sell'] || [],
            theme.palette.success.main,
            theme.palette.error.main
          );

          // Convert indicator overlays
          const timestamps = sortedData.map((x) => x['timestamp']['epoch_time']);
          const indicatorLines = indicators
            ? convertIndicatorData(indicators, timestamps)
            : [];

          setTradingViewData({
            ohlcvData,
            markers,
            indicators: indicatorLines,
          });
        } else {
          // Original Highcharts format
          const ohlcData = sortedData.map((x) => [
            x['timestamp']['epoch_time'] * 1000,
            x['open'],
            x['high'],
            x['low'],
            x['close'],
          ]);

          const options = {
            rangeSelector: {
              selected: 1,
            },
            series: [
              {
                id: 'dataseries',
                type: 'candlestick',
                data: ohlcData,
                color: theme.palette.secondary.light,
              },
              {
                type: 'flags',
                data: data['buy'].sort((a, b) => a['x'] - b['x']),
                onSeries: 'dataseries',
                shape: 'squarepin',
                width: 16,
                color: theme.palette.info.main,
                style: {
                  color: theme.palette.info.main,
                },
              },
              {
                type: 'flags',
                data: data['sell'].sort((a, b) => a['x'] - b['x']),
                onSeries: 'dataseries',
                shape: 'circlepin',
                width: 16,
                color: theme.palette.error.main,
                style: {
                  color: theme.palette.error.main,
                },
              },
            ],
          };

          if (indicators) {
            for (let indicator in indicators) {
              for (let line in indicators[indicator]) {
                options['series'].push({
                  name: [indicators[indicator][line]['label'], indicators[indicator][line]['name']].join(' - '),
                  data: ohlcData.map((e, i) => [e[0], indicators[indicator][line]['x'][i]]),
                });
              }
            }
          }

          setChart(
            <HighchartsReact
              constructorType={'stockChart'}
              highcharts={Highcharts}
              options={options}
            />
          );
        }
      });
  }, [data, provider, symbol, startDate, endDate, theme]);

  // Render TradingView chart if feature flag is enabled
  if (FEATURES.USE_TRADINGVIEW) {
    return (
      <React.Fragment>
        {tradingViewData.ohlcvData.length > 0 && (
          <TradingChart
            ohlcvData={tradingViewData.ohlcvData}
            markers={tradingViewData.markers}
            indicators={tradingViewData.indicators}
            height={400}
            upColor={theme.palette.success.main}
            downColor={theme.palette.error.main}
          />
        )}
      </React.Fragment>
    );
  }

  // Original Highcharts render
  return <React.Fragment>{chart}</React.Fragment>;
}
