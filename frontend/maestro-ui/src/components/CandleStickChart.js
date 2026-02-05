import React, { useEffect, useState } from 'react';
import { useTheme } from '@mui/material/styles';
import Title from './Title';
import Highcharts from 'highcharts/highstock';
import HighchartsReact from 'highcharts-react-official';

export default function CandleStickChart() {
  const theme = useTheme();
  const [chart, setChart] = useState(null);

  useEffect(() => {
    console.log('REACT_APP_REST_API_URL -> ', process.env.REACT_APP_REST_API_URL);
    fetch(`${process.env.REACT_APP_REST_API_URL}/datasource/bitmex/xbtusd/1d`)
      .then((res) => res.json())
      .then((data) => {
        const ohlcData = data['data']
          .sort((a, b) => a['timestamp']['epoch_time'] - b['timestamp']['epoch_time'])
          .map((x) => [
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
      })
      .catch(console.log);
  }, [theme.palette.secondary.light]);

  return (
    <React.Fragment>
      <Title>XBTUSD - Daily Data</Title>
      {chart}
    </React.Fragment>
  );
}
