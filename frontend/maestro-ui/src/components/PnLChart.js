import React, { useEffect, useState } from 'react';
import { useTheme } from '@mui/material/styles';
import Highcharts from 'highcharts/highstock';
import HighchartsReact from 'highcharts-react-official';

export default function PnLChart({ tid, data }) {
  const theme = useTheme();
  const [chart, setChart] = useState(null);

  useEffect(() => {
    if (!data) return;

    const options = {
      chart: {
        type: 'line',
      },
      title: null,
      subTitle: null,
      xAxis: {
        title: {
          text: 'Trade',
        },
      },
      yAxis: {
        title: {
          text: 'PNL',
        },
      },
      series: [
        {
          name: 'Pnl',
          color: theme.palette.secondary.light,
          data: data,
        },
      ],
    };

    setChart(<HighchartsReact highcharts={Highcharts} options={options} />);
  }, [tid, data, theme.palette.secondary.light]);

  return <React.Fragment>{chart}</React.Fragment>;
}
