import React, { useState, useEffect, useCallback } from 'react';
import Title from './Title';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import Grid from '@mui/material/Grid';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import CircularProgress from '@mui/material/CircularProgress';
import OptimizationProgress from './OptimizationProgress';
import TextField from '@mui/material/TextField';
import InputAdornment from '@mui/material/InputAdornment';
import Tooltip from '@mui/material/Tooltip';
import Skeleton from '@mui/material/Skeleton';
import Typography from '@mui/material/Typography';
import { DateTimePicker } from '@mui/x-date-pickers/DateTimePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { useOptimizationProgress } from '../hooks/useOptimizationProgress';
import { useNotification } from '../context/NotificationContext';
import SymbolChipSelector from './SymbolChipSelector';
import StrategyParamEditor from './StrategyParamEditor';
import TimeframeSelector from './TimeframeSelector';
import OptimizationTypeSelector from './OptimizationTypeSelector';

const formControlSx = { m: 0.5, minWidth: 120 };

export default function OptimizationForm({
  provider,
  symbol,
  binSize,
  providers,
  symbols,
  loadingProviders,
  loadingSymbols,
  onProviderChange,
  onSymbolChange,
  onBinSizeChange,
}) {
  const [testName, setTestName] = useState('Test Name');
  const [strategy, setStrategy] = useState('');
  const [strategies, setStrategies] = useState([]);
  const [optType, setOptType] = useState('BACKTESTING');
  const [isRunning, setIsRunning] = useState(false);
  const [runningTid, setRunningTid] = useState(null);
  const [isError, setIsError] = useState(false);
  const [testNameHelperText, setTestNameHelperText] = useState('');
  const [cash, setCash] = useState(10000);
  const [commissions, setCommissions] = useState(0.01);
  const [paramValues, setParamValues] = useState({});
  const [startDate, setStartDate] = useState(new Date(2000, 1, 1));
  const [endDate, setEndDate] = useState(new Date());
  const [loadingStrategies, setLoadingStrategies] = useState(true);

  // Centralized notification system
  const { notify } = useNotification();

  // Use WebSocket progress hook with polling fallback
  const { progress, isConnected, isComplete, error: progressError, status } = useOptimizationProgress(
    runningTid,
    isRunning
  );

  const updateStrategyParams = useCallback((strategyName) => {
    if (!strategyName) return;
    fetch(`${process.env.REACT_APP_REST_API_URL}/strategy/${strategyName}/params`)
      .then((response) => response.json())
      .then((data) => {
        setParamValues(data);
      })
      .catch((e) => notify(`Failed to load strategy params: ${e}`, 'error'));
  }, [notify]);

  useEffect(() => {
    setLoadingStrategies(true);
    fetch(`${process.env.REACT_APP_REST_API_URL}/strategy/available`)
      .then((response) => response.json())
      .then((data) => {
        setStrategies(data);
        if (data.length > 0) {
          setStrategy(data[0]);
          updateStrategyParams(data[0]);
        }
        setLoadingStrategies(false);
      })
      .catch((e) => {
        notify(`Failed to load strategies: ${e}`, 'error');
        setLoadingStrategies(false);
      });
  }, [updateStrategyParams, notify]);

  // Handle optimization completion
  useEffect(() => {
    if (isComplete && isRunning) {
      setIsRunning(false);
      if (progressError) {
        notify(`Optimization failed: ${progressError}`, 'error');
      } else if (status === 'completed') {
        notify('Optimization completed!', 'success');
      }
    }
  }, [isComplete, isRunning, progressError, status, notify]);

  const handleProviderChange = (e) => {
    onProviderChange(e.target.value);
  };

  const handleStrategyChange = (e) => {
    const value = e.target.value;
    setStrategy(value);
    updateStrategyParams(value);
  };

  const handleParamChange = (paramName, value) => {
    setParamValues((prev) => ({
      ...prev,
      [paramName]: value,
    }));
  };

  const submitTest = () => {
    const params = {
      test_name: testName,
      symbol: symbol,
      provider: provider,
      bin_size: binSize,
      strategy: strategy,
      kind: optType,
      cash: cash,
      commissions: commissions,
      start_date: startDate.getTime(),
      end_date: endDate.getTime(),
      strategy_params: { ...paramValues },
    };

    const requestOptions = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    };

    fetch(`${process.env.REACT_APP_REST_API_URL}/optimization/new/`, requestOptions)
      .then((response) => {
        const statusCode = response.status;
        const data = response.json();
        return Promise.all([statusCode, data]);
      })
      .then(([statusCode, data]) => {
        if (statusCode !== 200) {
          setIsError(true);
          setTestNameHelperText(data['error']);
          setIsRunning(false);
          setRunningTid(null);
        } else {
          // Start tracking via WebSocket hook
          setRunningTid(data['tid']);
          setIsRunning(true);
        }
      })
      .catch((e) => {
        notify(`Failed to start optimization: ${e}`, 'error');
        setIsRunning(false);
        setRunningTid(null);
      });
  };

  return (
    <LocalizationProvider dateAdapter={AdapterDateFns}>
      <React.Fragment>
        <Title>New Test</Title>
        <FormControl sx={formControlSx}>
          <TextField
            error={isError}
            id="test-name-text"
            label="Test Name"
            value={testName}
            helperText={testNameHelperText}
            variant="outlined"
            onChange={(e) => {
              setTestName(e.target.value);
              setIsError(false);
              setTestNameHelperText('');
            }}
          />
        </FormControl>
        <FormControl sx={formControlSx}>
          <DateTimePicker
            label="From"
            value={startDate}
            onChange={(d) => setStartDate(d)}
            ampm={false}
            format="yyyy/MM/dd HH:mm"
            disableFuture
          />
        </FormControl>
        <FormControl sx={formControlSx}>
          <DateTimePicker
            label="To"
            value={endDate}
            onChange={(d) => setEndDate(d)}
            ampm={false}
            format="yyyy/MM/dd HH:mm"
            disableFuture
          />
        </FormControl>
        <FormControl sx={formControlSx}>
          {loadingProviders ? (
            <Skeleton variant="rectangular" height={56} />
          ) : (
            <>
              <InputLabel id="provider-select-label">Provider</InputLabel>
              <Select
                labelId="provider-select-label"
                id="provider-select"
                value={provider}
                label="Provider"
                onChange={handleProviderChange}
              >
                {providers.map((p) => (
                  <MenuItem key={p} value={p}>
                    {p}
                  </MenuItem>
                ))}
              </Select>
            </>
          )}
        </FormControl>
        <Box sx={{ m: 0.5, minWidth: 200 }}>
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{ mb: 1, fontWeight: 500 }}
          >
            Symbol
          </Typography>
          <SymbolChipSelector
            provider={provider}
            value={symbol}
            onChange={onSymbolChange}
            disabled={!provider}
            symbols={symbols}
            loading={loadingSymbols}
          />
        </Box>
        <Box sx={{ m: 0.5, minWidth: 200 }}>
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{ mb: 1, fontWeight: 500 }}
          >
            Timeframe
          </Typography>
          <TimeframeSelector
            value={binSize}
            onChange={onBinSizeChange}
            disabled={!provider || !symbol}
          />
        </Box>
        <FormControl sx={formControlSx}>
          <TextField
            id="cash-text"
            label="Cash"
            type="number"
            value={cash}
            onChange={(e) => setCash(Number(e.target.value))}
            helperText="Initial portfolio value for simulation"
            InputProps={{
              startAdornment: <InputAdornment position="start">$</InputAdornment>,
            }}
          />
        </FormControl>
        <Tooltip title="Commission fee applied per trade. Enter as percentage (e.g., 0.1 = 0.1%)" arrow>
          <FormControl sx={formControlSx}>
            <TextField
              id="commissions-text"
              label="Commissions"
              type="number"
              value={commissions}
              onChange={(e) => setCommissions(Number(e.target.value))}
              helperText="Typical: 0.05-0.20% for crypto exchanges"
              InputProps={{
                endAdornment: <InputAdornment position="end">%</InputAdornment>,
              }}
            />
          </FormControl>
        </Tooltip>
        <FormControl sx={formControlSx}>
          {loadingStrategies ? (
            <Skeleton variant="rectangular" height={56} />
          ) : (
            <>
              <InputLabel id="strategy-select-label">Strategy</InputLabel>
              <Select
                labelId="strategy-select-label"
                id="strategy-select"
                value={strategy}
                label="Strategy"
                onChange={handleStrategyChange}
              >
                {strategies.map((row) => (
                  <MenuItem key={row} value={row}>
                    {row}
                  </MenuItem>
                ))}
              </Select>
            </>
          )}
        </FormControl>
        {/* Strategy Parameters - Using enhanced StrategyParamEditor */}
        {strategy && (
          <Box sx={{ m: 0.5, minWidth: 300 }}>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1, fontWeight: 500 }}>
              Strategy Parameters
            </Typography>
            <StrategyParamEditor
              strategy={strategy}
              values={paramValues}
              onChange={handleParamChange}
              showSliders={true}
            />
          </Box>
        )}
        <Box sx={{ m: 0.5, minWidth: 300 }}>
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{ mb: 1, fontWeight: 500 }}
          >
            Optimization Type
          </Typography>
          <OptimizationTypeSelector
            value={optType}
            onChange={setOptType}
          />
        </Box>
        <Grid container direction="row" justifyContent="flex-end" alignItems="center">
          <Box sx={{ m: 0.5 }}>
            <Button
              variant="contained"
              color="primary"
              onClick={submitTest}
              disabled={!provider || !symbol || !strategy || isRunning}
              startIcon={
                isRunning ? (
                  <CircularProgress size={20} color="inherit" />
                ) : (
                  <PlayArrowIcon />
                )
              }
            >
              {isRunning ? 'Running...' : 'Run Optimization'}
            </Button>
          </Box>
        </Grid>
        {isRunning && (
          <OptimizationProgress
            tid={runningTid}
            progress={progress}
            status={status}
            isConnected={isConnected}
            isComplete={isComplete}
            error={progressError}
          />
        )}
      </React.Fragment>
    </LocalizationProvider>
  );
}
