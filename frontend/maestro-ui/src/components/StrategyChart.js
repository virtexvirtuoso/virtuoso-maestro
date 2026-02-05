import React, { useEffect, useState } from 'react';
import { useTheme } from '@mui/material/styles';
import Highcharts from 'highcharts/highstock';
import HighchartsReact from 'highcharts-react-official';

export default function StrategyChart({ data, provider, symbol, startDate, endDate }) {
  const theme = useTheme();
  const [chart, setChart] = useState(null);

  useEffect(() => {
    if (!data || !provider || !symbol) return;

    const indicators = data['indicators'];
    fetch(
      `${process.env.REACT_APP_REST_API_URL}/datasource/${provider}/${symbol}/1d/${startDate}/${endDate}`
    )
      .then((res) => res.json())
      .then((responseData) => {
        const ohlcData = responseData['data']
          .sort((a, b) => a['timestamp']['epoch_time'] - b['timestamp']['epoch_time'])
          .map((x) => [
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
      });
  }, [data, provider, symbol, startDate, endDate, theme]);

  return <React.Fragment>{chart}</React.Fragment>;
}
