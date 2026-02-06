import React, { useEffect, useState } from 'react';
import { useTheme } from '@mui/material/styles';
import Title from './Title';
import Highcharts from 'highcharts/highstock';
import HighchartsReact from 'highcharts-react-official';
import TradingChart, { convertOHLCVData } from './TradingChart';
import { FEATURES } from '../config/featureFlags';

export default function CandleStickChart() {
  const theme = useTheme();
  const [chart, setChart] = useState(null);
  const [ohlcvData, setOhlcvData] = useState([]);

  useEffect(() => {
    console.log('REACT_APP_REST_API_URL -> ', process.env.REACT_APP_REST_API_URL);
    fetch(`${process.env.REACT_APP_REST_API_URL}/datasource/bitmex/xbtusd/1d`)
      .then((res) => res.json())
      .then((data) => {
        const sortedData = data['data'].sort(
          (a, b) => a['timestamp']['epoch_time'] - b['timestamp']['epoch_time']
        );

        if (FEATURES.USE_TRADINGVIEW) {
          // Convert data for TradingView Lightweight Charts
          setOhlcvData(convertOHLCVData(sortedData));
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
            series: [
              {
                type: 'candlestick',
                name: 'XBTUSD',
                color: theme.palette.secondary.light,
                data: ohlcData,
              },
            ],
            rangeSelector: {
              selected: 1,
            },
          };

          setChart(
            <HighchartsReact
              constructorType={'stockChart'}
              highcharts={Highcharts}
              options={options}
            />
          );
        }
      })
      .catch(console.log);
  }, [theme.palette.secondary.light]);

  // Render TradingView chart if feature flag is enabled
  if (FEATURES.USE_TRADINGVIEW) {
    return (
      <React.Fragment>
        <Title>XBTUSD - Daily Data</Title>
        {ohlcvData.length > 0 && (
          <TradingChart
            ohlcvData={ohlcvData}
            title=""
            height={400}
            upColor={theme.palette.success.main}
            downColor={theme.palette.error.main}
          />
        )}
      </React.Fragment>
    );
  }

  // Original Highcharts render
  return (
    <React.Fragment>
      <Title>XBTUSD - Daily Data</Title>
      {chart}
    </React.Fragment>
  );
}
