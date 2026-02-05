import React, { useState, useEffect, useCallback } from 'react';
import TableContainer from '@mui/material/TableContainer';
import Table from '@mui/material/Table';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import TableCell from '@mui/material/TableCell';
import TableBody from '@mui/material/TableBody';
import IconButton from '@mui/material/IconButton';
import KeyboardArrowUpIcon from '@mui/icons-material/KeyboardArrowUp';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Grid from '@mui/material/Grid';
import Collapse from '@mui/material/Collapse';
import ParametersDistributionPlot from './ParametersDistributionPlot';

const PERC_METRICS = new Set([
  'Annual return',
  'Annual volatility',
  'Cumulative returns',
  'Max drawdown',
  'Daily value at risk',
  'Daily turnover',
]);

const computeMean = (data) => {
  const sum = data.reduce((acc, value) => acc + value, 0);
  return sum / data.length;
};

const computeStd = (values, mean) => {
  const squareDiffs = values.map((value) => {
    const diff = value - mean;
    return diff * diff;
  });
  const meanSquareDiff = computeMean(squareDiffs);
  return Math.sqrt(meanSquareDiff);
};

const computeWalkForwardsStats = (datumElement) => {
  const metricArrays = {};

  for (let i = 0; i < datumElement.length; i++) {
    for (const key in datumElement[i]['analyzers']['PyFolio']) {
      if (!metricArrays.hasOwnProperty(key)) {
        metricArrays[key] = [];
      }
      metricArrays[key].push(datumElement[i]['analyzers']['PyFolio'][key]);
    }
  }

  return Object.keys(metricArrays).map((k) => {
    const mean = computeMean(metricArrays[k]);
    const std = computeStd(metricArrays[k], mean);
    return [k, mean, std];
  });
};

const getMetricValue = (metricName, metricValue) => {
  if (metricValue === null) {
    return 'N/A';
  }
  return PERC_METRICS.has(metricName)
    ? `${(metricValue * 100.0).toFixed(3)} %`
    : metricValue.toFixed(3);
};

const getMetricDollars = (metric, value, cash) => {
  if (value === null) {
    return 'N/A';
  }
  const formatter = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  });
  return PERC_METRICS.has(metric) ? formatter.format(value * cash) : '';
};

export default function WalkForwardMetrics({ tid, data }) {
  const [stats, setStats] = useState([]);
  const [expandedMetrics, setExpandedMetrics] = useState({});

  useEffect(() => {
    if (
      !data ||
      Object.keys(data).length === 0 ||
      !data['optimizations'] ||
      data['optimizations']['WALKFORWARD'].length === 0
    ) {
      setStats([]);
      return;
    }

    const computedStats = computeWalkForwardsStats(data['optimizations']['WALKFORWARD']);
    setStats(computedStats);

    // Reset expanded state
    const initialExpanded = {};
    computedStats.forEach((stat) => {
      initialExpanded[stat[0]] = false;
    });
    setExpandedMetrics(initialExpanded);
  }, [data]);

  const toggleMetricDistribution = useCallback((metric) => {
    setExpandedMetrics((prev) => ({
      ...prev,
      [metric]: !prev[metric],
    }));
  }, []);

  const getWalkForwardMetricsDistribution = useCallback(
    (metric) => {
      if (!data || !data['optimizations'] || data['optimizations']['WALKFORWARD'].length === 0) {
        return <p>No Data Available.</p>;
      }

      const optimizations = data['optimizations'];
      const metrics = {};

      const walkForwardData = optimizations['WALKFORWARD'].sort((a, b) =>
        a['num_split'] < b['num_split'] ? -1 : a['num_split'] > b['num_split'] ? 1 : 0
      );

      for (let i = 0; i < walkForwardData.length; i++) {
        const analyzerDict = walkForwardData[i]['analyzers']['PyFolio'];

        Object.keys(analyzerDict)
          .filter((m) => m === metric)
          .forEach((a) => {
            if (!metrics.hasOwnProperty(a)) {
              metrics[a] = [];
            }
            metrics[a].push(analyzerDict[a]);
          });
      }

      return (
        <Grid container>
          {Object.keys(metrics).map((m) => (
            <Grid item xs={12} key={m}>
              <ParametersDistributionPlot
                chartTitle=""
                xLabel="Out of Sample"
                yLabel="Metric"
                vLineLabel="Backtesting"
                vLineValue={optimizations['BACKTESTING'][0]['analyzers']['PyFolio'][m]}
                data={metrics[m]}
              />
            </Grid>
          ))}
        </Grid>
      );
    },
    [data]
  );

  if (stats.length === 0) {
    return <p>No Data Available</p>;
  }

  return (
    <TableContainer>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell />
            <TableCell>Metric</TableCell>
            <TableCell>Value</TableCell>
            <TableCell>Std. Dev.</TableCell>
            <TableCell>$</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {stats.map((m) => (
            <React.Fragment key={m[0] + '-rf'}>
              <TableRow
                hover
                sx={{
                  '& > *': {
                    borderBottom: 'unset',
                  },
                }}
              >
                <TableCell>
                  <IconButton
                    aria-label="expand row"
                    size="small"
                    onClick={() => toggleMetricDistribution(m[0])}
                  >
                    {expandedMetrics[m[0]] ? <KeyboardArrowUpIcon /> : <KeyboardArrowDownIcon />}
                  </IconButton>
                </TableCell>
                <TableCell>{m[0]}</TableCell>
                <TableCell>{getMetricValue(m[0], m[1])}</TableCell>
                <TableCell>&#177; {m[2] !== null ? m[2].toFixed(3) : 'N/A'}</TableCell>
                <TableCell>{getMetricDollars(m[0], m[1], data['cash'])}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell sx={{ paddingBottom: 0, paddingTop: 0 }} colSpan={6}>
                  <Collapse in={expandedMetrics[m[0]]} timeout="auto" unmountOnExit>
                    <Box margin={1}>
                      <Typography variant="h6" gutterBottom component="div">
                        {m[0]} Distribution
                      </Typography>
                      {getWalkForwardMetricsDistribution(m[0])}
                    </Box>
                  </Collapse>
                </TableCell>
              </TableRow>
            </React.Fragment>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}
