// Strategy Comparison Dashboard - Side-by-side analysis of multiple strategies
import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Autocomplete from '@mui/material/Autocomplete';
import TextField from '@mui/material/TextField';
import Table from '@mui/material/Table';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableCell from '@mui/material/TableCell';
import TableRow from '@mui/material/TableRow';
import TableBody from '@mui/material/TableBody';
import Typography from '@mui/material/Typography';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import StarIcon from '@mui/icons-material/Star';
import Title from './Title';
import EmptyState from './EmptyState';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';

const paperSx = {
  p: 2,
  display: 'flex',
  overflow: 'auto',
  flexDirection: 'column',
};

// Metrics configuration: name, extractor function, higherIsBetter flag
const METRICS_CONFIG = [
  { name: 'Sharpe Ratio', key: 'Sharpe ratio', higherIsBetter: true },
  { name: 'Annual Return', key: 'Annual return', higherIsBetter: true, isPercent: true },
  { name: 'Max Drawdown', key: 'Max drawdown', higherIsBetter: false, isPercent: true },
  { name: 'Win Rate', key: 'win_rate', higherIsBetter: true, isPercent: true, fromTradeAnalyzer: true },
  { name: 'Profit Factor', key: 'profit_factor', higherIsBetter: true, fromTradeAnalyzer: true },
  { name: 'Calmar Ratio', key: 'Calmar ratio', higherIsBetter: true },
  { name: 'Sortino Ratio', key: 'Sortino ratio', higherIsBetter: true },
];

// Color palette for strategy chips
const STRATEGY_COLORS = [
  '#fbbf24', // amber (primary)
  '#06B6D4', // cyan (secondary)
  '#10b981', // emerald
  '#8b5cf6', // violet
  '#f43f5e', // rose
  '#f97316', // orange
  '#14b8a6', // teal
  '#a855f7', // purple
];

// Extract metrics from PyFolio analyzer
const extractPyFolioMetric = (result, metricKey) => {
  try {
    const backtesting = result?.optimizations?.BACKTESTING;
    if (!backtesting || backtesting.length === 0) return null;
    const pyfolio = backtesting[0]?.analyzers?.PyFolio;
    if (!pyfolio) return null;
    return pyfolio[metricKey] ?? null;
  } catch {
    return null;
  }
};

// Extract metrics from TradeAnalyzer
const extractTradeAnalyzerMetric = (result, metricKey) => {
  try {
    const backtesting = result?.optimizations?.BACKTESTING;
    if (!backtesting || backtesting.length === 0) return null;
    const tradeAnalyzer = backtesting[0]?.analyzers?.TradeAnalyzer;
    if (!tradeAnalyzer) return null;

    if (metricKey === 'win_rate') {
      const won = tradeAnalyzer?.won?.total ?? 0;
      const lost = tradeAnalyzer?.lost?.total ?? 0;
      const total = won + lost;
      return total > 0 ? won / total : null;
    }

    if (metricKey === 'profit_factor') {
      const grossProfit = tradeAnalyzer?.pnl?.gross?.total ?? 0;
      const grossLoss = Math.abs(tradeAnalyzer?.pnl?.net?.total ?? 0) - grossProfit;
      return grossLoss > 0 ? Math.abs(grossProfit / grossLoss) : null;
    }

    return null;
  } catch {
    return null;
  }
};

// Extract a metric value from result
const extractMetric = (result, metricConfig) => {
  if (metricConfig.fromTradeAnalyzer) {
    return extractTradeAnalyzerMetric(result, metricConfig.key);
  }
  return extractPyFolioMetric(result, metricConfig.key);
};

// Format metric value for display
const formatMetricValue = (value, isPercent) => {
  if (value === null || value === undefined || isNaN(value)) {
    return 'N/A';
  }
  if (isPercent) {
    return `${(value * 100).toFixed(2)}%`;
  }
  return value.toFixed(3);
};

export default function StrategyComparison() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [availableTests, setAvailableTests] = useState([]);
  const [selectedStrategies, setSelectedStrategies] = useState([]);
  const [strategyResults, setStrategyResults] = useState({});
  const [loading, setLoading] = useState(false);

  // Fetch available tests on mount
  useEffect(() => {
    fetch(`${process.env.REACT_APP_REST_API_URL}/optimization/results/available`)
      .then((response) => response.json())
      .then((available) => setAvailableTests(available))
      .catch((error) => console.error('Failed to fetch available tests:', error));
  }, []);

  // Initialize from URL params
  useEffect(() => {
    const tidsParam = searchParams.get('tids');
    if (tidsParam && availableTests.length > 0) {
      const tids = tidsParam.split(',').filter(Boolean);
      const strategies = tids
        .map((tid) => availableTests.find((t) => t.tid === tid))
        .filter(Boolean);
      setSelectedStrategies(strategies);
    }
  }, [searchParams, availableTests]);

  // Fetch results for selected strategies
  const fetchStrategyResult = useCallback(async (tid) => {
    try {
      const response = await fetch(
        `${process.env.REACT_APP_REST_API_URL}/optimization/results/${tid}`
      );
      const data = await response.json();
      return { tid, data };
    } catch (error) {
      console.error(`Failed to fetch results for ${tid}:`, error);
      return { tid, data: null };
    }
  }, []);

  // Update URL params when selection changes
  useEffect(() => {
    const tids = selectedStrategies.map((s) => s.tid).join(',');
    if (tids) {
      setSearchParams({ tids });
    } else {
      setSearchParams({});
    }
  }, [selectedStrategies, setSearchParams]);

  // Fetch all selected strategy results
  useEffect(() => {
    const fetchAllResults = async () => {
      if (selectedStrategies.length === 0) {
        setStrategyResults({});
        return;
      }

      setLoading(true);
      const newTids = selectedStrategies
        .filter((s) => !strategyResults[s.tid])
        .map((s) => s.tid);

      if (newTids.length > 0) {
        const results = await Promise.all(newTids.map(fetchStrategyResult));
        const newResults = { ...strategyResults };
        results.forEach(({ tid, data }) => {
          if (data) newResults[tid] = data;
        });
        setStrategyResults(newResults);
      }
      setLoading(false);
    };

    fetchAllResults();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedStrategies, fetchStrategyResult]);

  // Handle adding a strategy
  const handleAddStrategy = (event, newValue) => {
    if (newValue && !selectedStrategies.find((s) => s.tid === newValue.tid)) {
      setSelectedStrategies([...selectedStrategies, newValue]);
    }
  };

  // Handle removing a strategy
  const handleRemoveStrategy = (tid) => {
    setSelectedStrategies(selectedStrategies.filter((s) => s.tid !== tid));
    // Also remove from cached results
    const newResults = { ...strategyResults };
    delete newResults[tid];
    setStrategyResults(newResults);
  };

  // Get color for a strategy
  const getStrategyColor = (index) => STRATEGY_COLORS[index % STRATEGY_COLORS.length];

  // Calculate comparison data
  const comparisonData = useMemo(() => {
    return METRICS_CONFIG.map((metric) => {
      const values = selectedStrategies.map((strategy) => {
        const result = strategyResults[strategy.tid];
        const value = result ? extractMetric(result, metric) : null;
        return { tid: strategy.tid, value };
      });

      // Find best value
      const validValues = values.filter((v) => v.value !== null && !isNaN(v.value));
      let bestTid = null;
      if (validValues.length > 0) {
        if (metric.higherIsBetter) {
          bestTid = validValues.reduce((a, b) => (a.value > b.value ? a : b)).tid;
        } else {
          bestTid = validValues.reduce((a, b) => (a.value < b.value ? a : b)).tid;
        }
      }

      return {
        metric,
        values,
        bestTid,
      };
    });
  }, [selectedStrategies, strategyResults]);

  // Filter out already selected strategies from autocomplete
  const availableOptions = availableTests.filter(
    (test) => !selectedStrategies.find((s) => s.tid === test.tid)
  );

  return (
    <Grid container spacing={3}>
      {/* Strategy Selector */}
      <Grid item xs={12}>
        <Paper sx={paperSx}>
          <Title>Compare Strategies</Title>
          <Box sx={{ mb: 2 }}>
            <Autocomplete
              options={availableOptions}
              getOptionLabel={(option) => option.test_name || option.tid}
              onChange={handleAddStrategy}
              value={null}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Add Strategy"
                  placeholder="Search and select strategies to compare..."
                  variant="outlined"
                  size="small"
                />
              )}
              sx={{ maxWidth: 500 }}
            />
          </Box>

          {/* Selected Strategies Chips */}
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, minHeight: 40 }}>
            {selectedStrategies.map((strategy, index) => (
              <Chip
                key={strategy.tid}
                label={strategy.test_name || strategy.tid}
                onDelete={() => handleRemoveStrategy(strategy.tid)}
                sx={{
                  backgroundColor: getStrategyColor(index),
                  color: '#000000',
                  fontWeight: 600,
                  '& .MuiChip-deleteIcon': {
                    color: '#000000',
                    '&:hover': {
                      color: '#333333',
                    },
                  },
                }}
              />
            ))}
            {selectedStrategies.length === 0 && (
              <Typography variant="body2" color="text.secondary">
                No strategies selected. Use the search above to add strategies.
              </Typography>
            )}
          </Box>
        </Paper>
      </Grid>

      {/* Comparison Table */}
      <Grid item xs={12}>
        <Paper sx={paperSx}>
          <Title>Metrics Comparison</Title>
          {selectedStrategies.length === 0 ? (
            <EmptyState
              icon={CompareArrowsIcon}
              title="No Strategies Selected"
              description="Add two or more strategies above to compare their performance metrics side by side."
            />
          ) : loading ? (
            <Typography variant="body2" color="text.secondary">
              Loading strategy data...
            </Typography>
          ) : (
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 700, minWidth: 150 }}>Metric</TableCell>
                    <TableCell sx={{ width: 40 }} />
                    {selectedStrategies.map((strategy, index) => (
                      <TableCell
                        key={strategy.tid}
                        align="center"
                        sx={{
                          fontWeight: 600,
                          borderBottom: `3px solid ${getStrategyColor(index)}`,
                          minWidth: 120,
                        }}
                      >
                        {strategy.test_name || strategy.tid}
                      </TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {comparisonData.map((row) => (
                    <TableRow key={row.metric.name} hover>
                      <TableCell sx={{ fontWeight: 500 }}>{row.metric.name}</TableCell>
                      <TableCell sx={{ px: 0.5 }}>
                        {row.metric.higherIsBetter ? (
                          <TrendingUpIcon
                            sx={{ fontSize: 16, color: 'success.main', opacity: 0.7 }}
                            titleAccess="Higher is better"
                          />
                        ) : (
                          <TrendingDownIcon
                            sx={{ fontSize: 16, color: 'error.main', opacity: 0.7 }}
                            titleAccess="Lower is better"
                          />
                        )}
                      </TableCell>
                      {row.values.map((v, index) => {
                        const isBest = v.tid === row.bestTid && row.bestTid !== null;
                        return (
                          <TableCell
                            key={v.tid}
                            align="center"
                            sx={{
                              backgroundColor: isBest
                                ? 'rgba(251, 191, 36, 0.15)'
                                : 'transparent',
                              fontWeight: isBest ? 700 : 400,
                              color: isBest ? 'primary.main' : 'text.primary',
                              position: 'relative',
                            }}
                          >
                            <Box
                              sx={{
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                gap: 0.5,
                              }}
                            >
                              {formatMetricValue(v.value, row.metric.isPercent)}
                              {isBest && (
                                <StarIcon
                                  sx={{
                                    fontSize: 14,
                                    color: 'primary.main',
                                  }}
                                />
                              )}
                            </Box>
                          </TableCell>
                        );
                      })}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </Paper>
      </Grid>
    </Grid>
  );
}
