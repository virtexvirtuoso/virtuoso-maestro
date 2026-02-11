import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useParams } from 'react-router-dom';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import Box from '@mui/material/Box';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Chip from '@mui/material/Chip';
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
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
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import ShowChartIcon from '@mui/icons-material/ShowChart';
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
import OptunaVisualization from './OptunaVisualization';
import QuantStatsReport from './QuantStatsReport';
import MonthlyReturnsHeatmap from './MonthlyReturnsHeatmap';
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

// Section tabs configuration
const SECTION_TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'charts', label: 'Charts' },
  { id: 'metrics', label: 'Metrics' },
  { id: 'walkforward', label: 'Walk-Forward' },
  { id: 'analysis', label: 'Analysis' },
];

// Summary metric card component
function MetricCard({ label, value, icon: Icon, trend, color }) {
  return (
    <Paper
      elevation={0}
      sx={{
        p: 2,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        bgcolor: 'background.paper',
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 2,
        minWidth: 140,
        flex: 1,
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
        {Icon && <Icon sx={{ fontSize: 18, color: 'text.secondary' }} />}
        <Box
          component="span"
          sx={{
            fontSize: '0.75rem',
            color: 'text.secondary',
            textTransform: 'uppercase',
            letterSpacing: 0.5,
          }}
        >
          {label}
        </Box>
      </Box>
      <Box
        sx={{
          fontSize: '1.5rem',
          fontWeight: 700,
          fontFamily: '"IBM Plex Mono", monospace',
          color: color || 'text.primary',
        }}
      >
        {value}
      </Box>
      {trend !== undefined && (
        <Chip
          size="small"
          icon={trend >= 0 ? <TrendingUpIcon /> : <TrendingDownIcon />}
          label={`${trend >= 0 ? '+' : ''}${(trend * 100).toFixed(1)}%`}
          color={trend >= 0 ? 'success' : 'error'}
          sx={{ mt: 0.5, height: 20, fontSize: '0.7rem' }}
        />
      )}
    </Paper>
  );
}

const paperSx = {
  p: 2,
  display: 'flex',
  overflow: 'auto',
  flexDirection: 'column',
};

// Collapsible section component with localStorage persistence
function CollapsibleSection({ id, title, subtitle, defaultExpanded = false, children }) {
  const storageKey = `eval_section_${id}`;
  const [expanded, setExpanded] = useState(() => {
    const stored = localStorage.getItem(storageKey);
    return stored !== null ? stored === 'true' : defaultExpanded;
  });

  const handleChange = (event, isExpanded) => {
    setExpanded(isExpanded);
    localStorage.setItem(storageKey, isExpanded.toString());
  };

  return (
    <Accordion
      expanded={expanded}
      onChange={handleChange}
      sx={{
        '&:before': { display: 'none' },
        bgcolor: 'background.paper',
        boxShadow: 1,
      }}
    >
      <AccordionSummary
        expandIcon={<ExpandMoreIcon />}
        sx={{
          '& .MuiAccordionSummary-content': {
            alignItems: 'center',
            gap: 2,
          },
        }}
      >
        <Box sx={{ fontWeight: 600, fontSize: '1rem' }}>{title}</Box>
        {subtitle && !expanded && (
          <Chip
            label={subtitle}
            size="small"
            sx={{ ml: 'auto', mr: 1, bgcolor: 'action.hover' }}
          />
        )}
      </AccordionSummary>
      <AccordionDetails sx={{ pt: 0 }}>{children}</AccordionDetails>
    </Accordion>
  );
}

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
  const [activeTab, setActiveTab] = useState('overview');

  // Extract summary metrics from results
  const summaryMetrics = useMemo(() => {
    if (Object.keys(results).length === 0) return null;

    try {
      const backtesting = results?.optimizations?.BACKTESTING;
      const pyfolio = backtesting?.[0]?.analyzers?.PyFolio;

      if (!pyfolio) return null;

      return {
        sharpe: pyfolio['Sharpe ratio'],
        annualReturn: pyfolio['Annual return'],
        maxDrawdown: pyfolio['Max drawdown'],
        volatility: pyfolio['Annual volatility'],
        calmar: pyfolio['Calmar ratio'],
        sortino: pyfolio['Sortino ratio'],
      };
    } catch {
      return null;
    }
  }, [results]);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const fetchResults = useCallback((testId) => {
    if (!testId) return;
    fetch(`${process.env.REACT_APP_REST_API_URL}/optimization/results/${testId}`)
      .then((response) => response.json())
      .then((data) => setResults(data))
      .catch((error) => console.log(error));
  }, []);

  useEffect(() => {
    fetch(`${process.env.REACT_APP_REST_API_URL}/optimization/results`)
      .then((response) => response.json())
      .then((data) => {
        // Ensure we always have an array
        const available = Array.isArray(data) ? data : [];
        setAvailableTests(available);
        if (paramTid) {
          setTid(paramTid);
          fetchResults(paramTid);
        }
      })
      .catch((error) => {
        console.error('Failed to fetch available tests:', error);
        setAvailableTests([]);
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

  const getOptunaVisualization = () => {
    if (Object.keys(results).length === 0) {
      return <NoDataMsg />;
    }

    // Check if this is a V2 optimization result with Optuna study data
    const optunaStudy = results['optuna_study'] || results['study'];
    const optunaTrials = results['optuna_trials'] || results['trials'];

    if (!optunaStudy && !optunaTrials) {
      return (
        <p>
          No Optuna optimization data available. This section shows results from Optuna-based
          hyperparameter optimization studies (V2 API).
        </p>
      );
    }

    // Build study data object for OptunaVisualization
    const studyData = {
      study_name: optunaStudy?.name || results['test_name'],
      trials: optunaTrials || [],
    };

    return <OptunaVisualization studyData={studyData} />;
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

  const getQuantStatsReport = () => {
    if (Object.keys(results).length === 0) {
      return <NoDataMsg />;
    }

    // Try to get QuantStats data from various possible locations in the API response
    const quantStats = results['quantstats'] || results['quant_stats'] || results['metrics'];

    // Also check if metrics are nested in backtesting results
    const backtestData = results['optimizations']?.['BACKTESTING']?.[0];
    const pyfolioMetrics = backtestData?.['analyzers']?.['PyFolio'];

    // Build a combined metrics object from available data
    if (!quantStats && !pyfolioMetrics) {
      return <p>QuantStats not available. Run an optimization with QuantStats analyzer enabled.</p>;
    }

    // If we have PyFolio metrics, transform them to QuantStats-like format
    const metricsData = quantStats || {
      total_return: pyfolioMetrics?.['Cumulative returns'],
      cagr: pyfolioMetrics?.['Annual return'],
      volatility: pyfolioMetrics?.['Annual volatility'],
      max_drawdown: pyfolioMetrics?.['Max drawdown'],
      sharpe: pyfolioMetrics?.['Sharpe ratio'],
      sortino: pyfolioMetrics?.['Sortino ratio'],
      calmar: pyfolioMetrics?.['Calmar ratio'],
      daily_var: pyfolioMetrics?.['Daily value at risk'],
    };

    return <QuantStatsReport data={metricsData} />;
  };

  const getMonthlyReturnsHeatmap = () => {
    if (Object.keys(results).length === 0) {
      return <NoDataMsg />;
    }

    // Try to get monthly returns data from various possible locations
    const monthlyReturns = results['monthly_returns'] || results['monthlyReturns'];

    if (!monthlyReturns) {
      return <p>Monthly returns data not available. This requires QuantStats or monthly aggregation enabled.</p>;
    }

    return <MonthlyReturnsHeatmap data={monthlyReturns} />;
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
      {/* Test Selector */}
      <Grid item xs={12}>
        <Paper sx={paperSx}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 2 }}>
            <Title>Evaluation</Title>
            <FormControl sx={{ minWidth: 250 }} size="small">
              <InputLabel id="test-select-label">Select Test</InputLabel>
              <Select
                labelId="test-select-label"
                id="test-select"
                name="test"
                value={tid}
                label="Select Test"
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
          </Box>
        </Paper>
      </Grid>

      {/* Summary Metrics Row */}
      {summaryMetrics && (
        <Grid item xs={12}>
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            <MetricCard
              label="Sharpe Ratio"
              value={summaryMetrics.sharpe?.toFixed(2) ?? '—'}
              icon={ShowChartIcon}
              color={summaryMetrics.sharpe >= 1 ? 'success.main' : summaryMetrics.sharpe >= 0 ? 'warning.main' : 'error.main'}
            />
            <MetricCard
              label="Annual Return"
              value={summaryMetrics.annualReturn ? `${(summaryMetrics.annualReturn * 100).toFixed(1)}%` : '—'}
              icon={TrendingUpIcon}
              color={summaryMetrics.annualReturn >= 0 ? 'success.main' : 'error.main'}
            />
            <MetricCard
              label="Max Drawdown"
              value={summaryMetrics.maxDrawdown ? `${(summaryMetrics.maxDrawdown * 100).toFixed(1)}%` : '—'}
              icon={TrendingDownIcon}
              color={summaryMetrics.maxDrawdown > -0.1 ? 'success.main' : summaryMetrics.maxDrawdown > -0.2 ? 'warning.main' : 'error.main'}
            />
            <MetricCard
              label="Volatility"
              value={summaryMetrics.volatility ? `${(summaryMetrics.volatility * 100).toFixed(1)}%` : '—'}
              icon={ShowChartIcon}
            />
            <MetricCard
              label="Calmar Ratio"
              value={summaryMetrics.calmar?.toFixed(2) ?? '—'}
              icon={ShowChartIcon}
              color={summaryMetrics.calmar >= 1 ? 'success.main' : 'text.primary'}
            />
            <MetricCard
              label="Sortino Ratio"
              value={summaryMetrics.sortino?.toFixed(2) ?? '—'}
              icon={ShowChartIcon}
              color={summaryMetrics.sortino >= 1 ? 'success.main' : 'text.primary'}
            />
          </Box>
        </Grid>
      )}

      {/* Section Navigation Tabs */}
      <Grid item xs={12}>
        <Paper sx={{ ...paperSx, p: 0 }}>
          <Tabs
            value={activeTab}
            onChange={handleTabChange}
            variant="scrollable"
            scrollButtons="auto"
            sx={{
              borderBottom: 1,
              borderColor: 'divider',
              '& .MuiTab-root': {
                textTransform: 'none',
                fontWeight: 600,
                minWidth: 100,
              },
            }}
          >
            {SECTION_TABS.map((tab) => (
              <Tab key={tab.id} label={tab.label} value={tab.id} />
            ))}
          </Tabs>
        </Paper>
      </Grid>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <>
          <Grid item xs={12}>
            <Paper sx={paperSx}>
              <Title>Strategy Evaluation</Title>
              {getStrategyDetails()}
            </Paper>
          </Grid>
          <Grid item xs={6}>
            <Paper sx={paperSx}>
              <Title>Backtesting Metrics</Title>
              {getBacktestMetrics()}
            </Paper>
          </Grid>
          <Grid item xs={6}>
            <Paper sx={paperSx}>
              <Title>Walk Forward Metrics</Title>
              <WalkForwardMetrics tid={tid} data={results} />
            </Paper>
          </Grid>
        </>
      )}

      {/* Charts Tab */}
      {activeTab === 'charts' && (
        <Grid item xs={12}>
          <CollapsibleSection id="backtest-strategy" title="Backtest Strategy Chart" defaultExpanded={true}>
            {getBacktestStrategyChart()}
          </CollapsibleSection>
          <CollapsibleSection id="backtest-pnl" title="Backtest PnL Chart" subtitle="Profit/Loss">
            {getBacktestPnLChart()}
          </CollapsibleSection>
          <CollapsibleSection id="wfo-strategy" title="Walk-Forward Strategy Chart">
            {getWalkForwardStrategyChart()}
          </CollapsibleSection>
          <CollapsibleSection id="wfo-pnl" title="Walk-Forward PnL Chart" subtitle="Out-of-Sample">
            {getWalkForwardPnLChart()}
          </CollapsibleSection>
        </Grid>
      )}

      {/* Metrics Tab */}
      {activeTab === 'metrics' && (
        <Grid item xs={12}>
          <CollapsibleSection id="quantstats" title="QuantStats Report" defaultExpanded={true}>
            {getQuantStatsReport()}
          </CollapsibleSection>
          <CollapsibleSection id="monthly-returns" title="Monthly Returns Heatmap" subtitle="Performance by Month">
            {getMonthlyReturnsHeatmap()}
          </CollapsibleSection>
        </Grid>
      )}

      {/* Walk-Forward Tab */}
      {activeTab === 'walkforward' && (
        <Grid item xs={12}>
          <CollapsibleSection id="wfo-timeline" title="Walk-Forward Timeline" defaultExpanded={true} subtitle="Train/Test Splits">
            {getWalkForwardTimeline()}
          </CollapsibleSection>
          <CollapsibleSection id="param-stability" title="Parameter Stability" subtitle="Across Splits">
            {getParameterStabilityChart()}
          </CollapsibleSection>
        </Grid>
      )}

      {/* Analysis Tab */}
      {activeTab === 'analysis' && (
        <Grid item xs={12}>
          <CollapsibleSection id="param-dist" title="Parameters Distribution" defaultExpanded={true} subtitle="WFO vs Backtest">
            {getParametersDistribution()}
          </CollapsibleSection>
          <CollapsibleSection id="heatmap" title="Heatmap Chart" subtitle="Parameter Correlation">
            {getHeatMapChart()}
          </CollapsibleSection>
          <CollapsibleSection id="optuna" title="Optuna Optimization" subtitle="Hyperparameter Search">
            {getOptunaVisualization()}
          </CollapsibleSection>
        </Grid>
      )}
    </Grid>
  );
}
