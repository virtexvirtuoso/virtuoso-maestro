import React, { useEffect, useState } from 'react';
import { useTheme } from '@mui/material/styles';
import Highcharts from 'highcharts/highstock';
import HighchartsReact from 'highcharts-react-official';
import bellcurve from 'highcharts/modules/histogram-bellcurve';

bellcurve(Highcharts);

export default function ParametersDistributionPlot({
  tid,
  chartTitle,
  xLabel,
  yLabel,
  vLineLabel,
  vLineValue,
  data,
}) {
  const theme = useTheme();
  const [chart, setChart] = useState(null);

  useEffect(() => {
    if (!data) return;

    const options = {
      title: {
        text: chartTitle,
      },
      xAxis: [
        {
          title: {
            text: xLabel,
          },
          alignTicks: false,
        },
        {
          title: {
            text: 'Bell curve',
          },
          alignTicks: false,
          opposite: true,
          plotLines: [
            {
              label: vLineLabel,
              color: theme.palette.warning.main,
              width: 3,
              value: vLineValue,
            },
          ],
        },
      ],
      yAxis: [
        {
          title: { text: yLabel },
        },
        {
          title: { text: 'Bell curve' },
          opposite: true,
        },
      ],
      series: [
        {
          name: 'Bell curve',
          type: 'bellcurve',
          xAxis: 1,
          yAxis: 1,
          baseSeries: 1,
          zIndex: -10,
          opacity: 0.3,
          color: theme.palette.secondary.light,
        },
        {
          name: yLabel,
          type: 'scatter',
          data: data,
          color: theme.palette.primary.main,
          accessibility: {
            exposeAsGroupOnly: true,
          },
          marker: {
            radius: 3.5,
          },
        },
      ],
    };

    setChart(<HighchartsReact highcharts={Highcharts} options={options} />);
  }, [tid, chartTitle, xLabel, yLabel, vLineLabel, vLineValue, data, theme]);

  return <React.Fragment>{chart}</React.Fragment>;
}
