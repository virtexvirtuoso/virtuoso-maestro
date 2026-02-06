import React, { useState, useEffect, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import Title from './Title';
import Table from '@mui/material/Table';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableCell from '@mui/material/TableCell';
import TableRow from '@mui/material/TableRow';
import TableBody from '@mui/material/TableBody';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import List from '@mui/material/List';
import TextFieldsIcon from '@mui/icons-material/TextFields';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import ListItemAvatar from '@mui/material/ListItemAvatar';
import Avatar from '@mui/material/Avatar';
import MonetizationOnIcon from '@mui/icons-material/MonetizationOn';
import TimerIcon from '@mui/icons-material/Timer';
import StrategyChart from './StrategyChart';
import PnLChart from './PnLChart';
import AccountBalanceIcon from '@mui/icons-material/AccountBalance';
import MoneyOffIcon from '@mui/icons-material/MoneyOff';
import ParametersDistributionPlot from './ParametersDistributionPlot';
import IconButton from '@mui/material/IconButton';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import WalkForwardMetrics from './WalkForwardMetrics';
import HeatMapChart from './HeatMap';
import WalkForwardTimeline from './WalkForwardTimeline';
import ParameterStabilityChart from './ParameterStabilityChart';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import DateRangeIcon from '@mui/icons-material/DateRange';
import BusinessIcon from '@mui/icons-material/Business';
import Button from '@mui/material/Button';
import Link from '@mui/material/Link';
import EmptyState from './EmptyState';
import AssessmentIcon from '@mui/icons-material/Assessment';

const paperSx = {
  p: 2,
  display: 'flex',
  overflow: 'auto',
  flexDirection: 'column',
};

const formControlSx = { m: 0.5, minWidth: 120 };

const PERC_METRICS = new Set([
  'Annual return',
  'Annual volatility',
  'Cumulative returns',
  'Max drawdown',
  'Daily value at risk',
  'Daily turnover',
]);

const getMetricValue = (metric, pyfolioMetrics) => {
  if (pyfolioMetrics[metric] === null) {
    return 'N/A';
  }
  return PERC_METRICS.has(metric)
    ? `${(pyfolioMetrics[metric] * 100.0).toFixed(3)} %`
    : pyfolioMetrics[metric].toFixed(3);
};

const getMetricDollars = (metric, pyfolioMetrics, cash) => {
  if (pyfolioMetrics[metric] === null) {
    return 'N/A';
  }
  const formatter = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  });
  return PERC_METRICS.has(metric) ? formatter.format(pyfolioMetrics[metric] * cash) : '';
};

export default function Evaluation() {
  const { tid: paramTid } = useParams();
  const [tid, setTid] = useState('');
  const [availableTests, setAvailableTests] = useState([]);
  const [results, setResults] = useState({});

  const fetchResults = useCallback((testId) => {
    if (!testId) return;
    fetch(`${process.env.REACT_APP_REST_API_URL}/optimization/results/${testId}`)
      .then((response) => response.json())
      .then((data) => setResults(data))
      .catch((error) => console.log(error));
  }, []);

  useEffect(() => {
    fetch(`${process.env.REACT_APP_REST_API_URL}/optimization/results/available`)
      .then((response) => response.json())
      .then((available) => {
        setAvailableTests(available);
        if (paramTid) {
          setTid(paramTid);
          fetchResults(paramTid);
        }
      });
  }, [paramTid, fetchResults]);

  const handleInputChange = (event) => {
    const newTid = event.target.value;
    setTid(newTid);
    fetchResults(newTid);
  };

  const NoDataMsg = () => <p>No Data Available</p>;

  const getBacktestMetrics = () => {
    if (Object.keys(results).length === 0) {
      return <NoDataMsg />;
    }

    const datumElement = results['optimizations']['BACKTESTING'];

    if (datumElement.length === 0) {
      return <NoDataMsg />;
    }

    const pyfolioMetrics = datumElement[0]['analyzers']['PyFolio'];

    return (
      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell />
              <TableCell>Metric</TableCell>
              <TableCell>Value</TableCell>
              <TableCell>$</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {Object.keys(pyfolioMetrics).map((m) => (
              <TableRow key={m} hover>
                <TableCell>
                  <IconButton aria-label="expand row" size="small" disabled>
                    <KeyboardArrowDownIcon />
                  </IconButton>
                </TableCell>
                <TableCell>{m}</TableCell>
                <TableCell>{getMetricValue(m, pyfolioMetrics)}</TableCell>
                <TableCell>{getMetricDollars(m, pyfolioMetrics, results['cash'])}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    );
  };

  const getStrategyDetails = () => {
    const formatter = new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    });

    if (Object.keys(results).length === 0) {
      return <NoDataMsg />;
    }

    return (
      <List dense={true}>
        <ListItem>
          <ListItemAvatar>
            <Avatar>
              <TextFieldsIcon />
            </Avatar>
          </ListItemAvatar>
          <ListItemText primary={results['test_name']} secondary="Test Name" />
        </ListItem>
        <ListItem>
          <ListItemAvatar>
            <Avatar>
              <AccessTimeIcon />
            </Avatar>
          </ListItemAvatar>
          <ListItemText
            primary={new Date(results['creation_time']).toLocaleString()}
            secondary="Creation Time"
          />
        </ListItem>
        <ListItem>
          <ListItemAvatar>
            <Avatar>
              <DateRangeIcon />
            </Avatar>
          </ListItemAvatar>
          <ListItemText
            primary={
              <span>
                From <b> {new Date(results['start_date']).toLocaleString()} </b>
                To <b> {new Date(results['end_date']).toLocaleString()} </b>
              </span>
            }
            secondary="Time Interval"
          />
        </ListItem>
        <ListItem>
          <ListItemAvatar>
            <Avatar>
              <BusinessIcon />
            </Avatar>
          </ListItemAvatar>
          <ListItemText primary={results['provider']} secondary="Provider" />
        </ListItem>
        <ListItem>
          <ListItemAvatar>
            <Avatar>
              <AccountBalanceIcon />
            </Avatar>
          </ListItemAvatar>
          <ListItemText primary={results['symbol']} secondary="Symbol" />
        </ListItem>
        <ListItem>
          <ListItemAvatar>
            <Avatar>
              <TimerIcon />
            </Avatar>
          </ListItemAvatar>
          <ListItemText primary={results['timeframe']} secondary="Timeframe" />
        </ListItem>
        <ListItem>
          <ListItemAvatar>
            <Avatar>
              <MonetizationOnIcon />
            </Avatar>
          </ListItemAvatar>
          <ListItemText primary={formatter.format(results['cash'])} secondary="Cash" />
        </ListItem>
        <ListItem>
          <ListItemAvatar>
            <Avatar>
              <MoneyOffIcon />
            </Avatar>
          </ListItemAvatar>
          <ListItemText primary={results['commissions'] + '%'} secondary="Commissions" />
        </ListItem>
        <ListItem>
          <Link
            href={`${process.env.REACT_APP_REST_API_URL}/optimization/results/${tid}/report`}
            target="_blank"
            rel="noopener noreferrer"
            download
            underline="none"
          >
            <Button variant="contained" color="primary" component="span">
              Export
            </Button>
          </Link>
        </ListItem>
      </List>
    );
  };

  const getBacktestStrategyChart = () => {
    if (Object.keys(results).length === 0) {
      return <NoDataMsg />;
    }

    const datumElement = results['optimizations']['BACKTESTING'];

    if (datumElement.length === 0) {
      return <NoDataMsg />;
    }

    const buy = datumElement[0]['observers']['BuySell']['buy'].map((e) => ({
      x: e[0],
      title: e[1].toString(),
      text: `Entry at ${e[1]}`,
    }));

    const sell = datumElement[0]['observers']['BuySell']['sell'].map((e) => ({
      x: e[0],
      title: e[1].toString(),
      text: `Stopped at ${e[1]}`,
    }));

    const indicators = [];
    for (let indicator in datumElement[0]['indicators']) {
      const lines = [];
      for (let line in datumElement[0]['indicators'][indicator]) {
        const plotline = {
          name: line,
          label: indicator,
          x: datumElement[0]['indicators'][indicator][line]['values'],
        };
        lines.push(plotline);
      }
      indicators.push(lines);
    }

    const data = {
      buy: buy,
      sell: sell,
      indicators: indicators,
    };

    return (
      <StrategyChart
        provider={results['provider']}
        symbol={results['symbol']}
        startDate={results['start_date']}
        endDate={results['end_date']}
        data={data}
      />
    );
  };

  const getWalkForwardStrategyChart = () => {
    if (Object.keys(results).length === 0) {
      return <NoDataMsg />;
    }

    const datumElement = results['optimizations']['WALKFORWARD'];

    if (datumElement.length === 0) {
      return <NoDataMsg />;
    }

    let buy = [];
    let sell = [];

    for (let i = 0; i < datumElement.length; i++) {
      const buyTmp = datumElement[i]['observers']['BuySell']['buy'].map((e) => ({
        x: e[0],
        title: e[1].toString(),
        text: `Entry at ${e[1]} (Num Split: ${i})`,
      }));

      const sellTmp = datumElement[i]['observers']['BuySell']['sell'].map((e) => ({
        x: e[0],
        title: e[1].toString(),
        text: `Stopped at ${e[1]} (Num Split: ${i})`,
      }));

      buy = buy.concat(buyTmp);
      sell = sell.concat(sellTmp);
    }

    const data = {
      buy: buy,
      sell: sell,
    };

    return (
      <StrategyChart
        provider={results['provider']}
        symbol={results['symbol']}
        startDate={results['start_date']}
        endDate={results['end_date']}
        data={data}
      />
    );
  };

  const getBacktestPnLChart = () => {
    if (Object.keys(results).length === 0) {
      return <NoDataMsg />;
    }

    const datumElement = results['optimizations']['BACKTESTING'];

    if (datumElement.length === 0) {
      return <NoDataMsg />;
    }

    return <PnLChart tid={results['tid']} data={datumElement[0]['observers']['Trades']['pnl']} />;
  };

  const getWalkForwardPnLChart = () => {
    if (Object.keys(results).length === 0) {
      return <NoDataMsg />;
    }

    const datumElement = results['optimizations']['WALKFORWARD'];

    if (datumElement.length === 0) {
      return <NoDataMsg />;
    }

    let pnl = [];

    for (let i = 0; i < datumElement.length; i++) {
      pnl = pnl.concat(datumElement[i]['observers']['Trades']['pnl']);
    }

    return <PnLChart tid={results['tid']} data={pnl} />;
  };

  const getParametersDistribution = () => {
    if (Object.keys(results).length === 0) {
      return <NoDataMsg />;
    }

    const optimizations = results['optimizations'];

    if (
      optimizations['BACKTESTING'].length === 0 ||
      optimizations['WALKFORWARD'].length === 0 ||
      Object.keys(results['parameters']).length === 0
    ) {
      return <p>Current Strategy does not have any custom parameters.</p>;
    }

    return (
      <Grid container>
        {Object.keys(results['parameters']).map((p) => (
          <Grid item xs={4} key={p}>
            <ParametersDistributionPlot
              tid={results['tid']}
              chartTitle={p.charAt(0).toUpperCase() + p.slice(1)}
              xLabel="Out of Sample"
              yLabel="Parameter"
              vLineLabel="Backtesting"
              vLineValue={results['optimizations']['BACKTESTING'][0]['parameters'][p]}
              data={results['optimizations']['WALKFORWARD']
                .sort((a, b) =>
                  a['num_split'] < b['num_split'] ? -1 : a['num_split'] > b['num_split'] ? 1 : 0
                )
                .map((r) => r['parameters'][p])}
            />
          </Grid>
        ))}
      </Grid>
    );
  };

  const getHeatMapChart = () => {
    if (Object.keys(results).length === 0) {
      return <NoDataMsg />;
    }

    const optimizations = results['optimizations'];
    if (optimizations['BACKTESTING'].length === 0 || optimizations['WALKFORWARD'].length === 0) {
      return (
        <p>
          No Data Available. You need to perform both Backtesting and Walkforward testing to get
          parameters distribution.
        </p>
      );
    }

    return <HeatMapChart tid={results['tid']} />;
  };

  const getWalkForwardTimeline = () => {
    if (Object.keys(results).length === 0) {
      return <NoDataMsg />;
    }

    const walkforwardData = results['optimizations']?.['WALKFORWARD'];
    if (!walkforwardData || walkforwardData.length === 0) {
      return <p>No walk-forward data available.</p>;
    }

    // Extract split timeline data - using available date fields from each split
    const splits = walkforwardData.map((split) => ({
      num_split: split.num_split,
      train_start: split.train_start || split.start_date,
      train_end: split.train_end || split.end_date,
      test_start: split.test_start,
      test_end: split.test_end || split.end_date,
      start_date: split.start_date,
      end_date: split.end_date,
    }));

    return (
      <WalkForwardTimeline
        splits={splits}
        startDate={results['start_date']}
        endDate={results['end_date']}
      />
    );
  };

  const getParameterStabilityChart = () => {
    if (Object.keys(results).length === 0) {
      return <NoDataMsg />;
    }

    const walkforwardData = results['optimizations']?.['WALKFORWARD'];
    if (!walkforwardData || walkforwardData.length === 0) {
      return <p>No walk-forward data available.</p>;
    }

    // Check if there are any parameters
    if (
      !results['parameters'] ||
      Object.keys(results['parameters']).length === 0
    ) {
      return <p>Current strategy does not have any custom parameters to analyze.</p>;
    }

    return (
      <ParameterStabilityChart
        walkforwardData={walkforwardData}
        parameterRanges={results['parameters']}
      />
    );
  };

  // Show EmptyState if no test is selected and no data loaded
  if (!tid && Object.keys(results).length === 0) {
    return (
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={paperSx}>
            <Title>Tests</Title>
            {availableTests.length === 0 ? (
              <EmptyState
                icon={AssessmentIcon}
                title="No Data Loaded"
                description="No optimization tests are available. Run a test first to see evaluation results."
              />
            ) : (
              <>
                <FormControl sx={formControlSx}>
                  <InputLabel id="test-select-label">Test</InputLabel>
                  <Select
                    labelId="test-select-label"
                    id="test-select"
                    name="test"
                    value={tid}
                    label="Test"
                    onChange={handleInputChange}
                  >
                    {availableTests
                      .sort((a, b) => (a['test_name'] < b['test_name'] ? -1 : 1))
                      .map((row) => (
                        <MenuItem key={row['tid']} value={row['tid']}>
                          {row['test_name']}
                        </MenuItem>
                      ))}
                  </Select>
                </FormControl>
                <EmptyState
                  icon={AssessmentIcon}
                  title="Select a Test"
                  description="Choose an optimization test from the dropdown above to view detailed evaluation results."
                />
              </>
            )}
          </Paper>
        </Grid>
      </Grid>
    );
  }

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Paper sx={paperSx}>
          <Title>Tests</Title>
          <FormControl sx={formControlSx}>
            <InputLabel id="test-select-label">Test</InputLabel>
            <Select
              labelId="test-select-label"
              id="test-select"
              name="test"
              value={tid}
              label="Test"
              onChange={handleInputChange}
            >
              {availableTests
                .sort((a, b) => (a['test_name'] < b['test_name'] ? -1 : 1))
                .map((row) => (
                  <MenuItem key={row['tid']} value={row['tid']}>
                    {row['test_name']}
                  </MenuItem>
                ))}
            </Select>
          </FormControl>
        </Paper>
      </Grid>
      <Grid item xs={12}>
        <Paper sx={paperSx}>
          <Title>Strategy Evaluation</Title>
          {getStrategyDetails()}
        </Paper>
      </Grid>
      <Grid item xs={12}>
        <Paper sx={paperSx}>
          <Title>Backtest Strategy Chart</Title>
          {getBacktestStrategyChart()}
        </Paper>
      </Grid>
      <Grid item xs={12}>
        <Paper sx={paperSx}>
          <Title>Walkforward Strategy Chart</Title>
          {getWalkForwardStrategyChart()}
        </Paper>
      </Grid>
      <Grid item xs={6}>
        <Paper sx={paperSx}>
          <Title>Backtesting</Title>
          {getBacktestMetrics()}
        </Paper>
      </Grid>
      <Grid item xs={6}>
        <Paper sx={paperSx}>
          <Title>Walk Forward</Title>
          <WalkForwardMetrics tid={tid} data={results} />
        </Paper>
      </Grid>
      <Grid item xs={12}>
        <Paper sx={paperSx}>
          <Title>Walk-Forward Timeline</Title>
          {getWalkForwardTimeline()}
        </Paper>
      </Grid>
      <Grid item xs={12}>
        <Paper sx={paperSx}>
          <Title>Parameter Stability</Title>
          {getParameterStabilityChart()}
        </Paper>
      </Grid>
      <Grid item xs={12}>
        <Paper sx={paperSx}>
          <Title>Backtest PnL Chart</Title>
          {getBacktestPnLChart()}
        </Paper>
      </Grid>
      <Grid item xs={12}>
        <Paper sx={paperSx}>
          <Title>Walkforward PnL Chart</Title>
          {getWalkForwardPnLChart()}
        </Paper>
      </Grid>
      <Grid item xs={12}>
        <Paper sx={paperSx}>
          <Title>Parameters Distribution</Title>
          {getParametersDistribution()}
        </Paper>
      </Grid>
      <Grid item xs={12}>
        <Paper sx={paperSx}>
          <Title>Heatmap Chart</Title>
          {getHeatMapChart()}
        </Paper>
      </Grid>
    </Grid>
  );
}
