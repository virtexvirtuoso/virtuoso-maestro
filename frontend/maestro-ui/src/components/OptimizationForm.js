import React, { useState, useEffect, useCallback } from 'react';
import Title from './Title';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import Grid from '@mui/material/Grid';
import Box from '@mui/material/Box';
import IconButton from '@mui/material/IconButton';
import PlayCircleFilledWhiteIcon from '@mui/icons-material/PlayCircleFilledWhite';
import OptimizationProgress from './OptimizationProgress';
import TextField from '@mui/material/TextField';
import InputAdornment from '@mui/material/InputAdornment';
import Tooltip from '@mui/material/Tooltip';
import FormLabel from '@mui/material/FormLabel';
import RadioGroup from '@mui/material/RadioGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import Radio from '@mui/material/Radio';
import Skeleton from '@mui/material/Skeleton';
import FormHelperText from '@mui/material/FormHelperText';
import { DateTimePicker } from '@mui/x-date-pickers/DateTimePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { useOptimizationProgress } from '../hooks/useOptimizationProgress';
import { useNotification } from '../context/NotificationContext';

const formControlSx = { m: 0.5, minWidth: 120 };

export default function OptimizationForm() {
  const [testName, setTestName] = useState('Test Name');
  const [provider, setProvider] = useState('');
  const [providers, setProviders] = useState([]);
  const [symbol, setSymbol] = useState('');
  const [symbols, setSymbols] = useState([]);
  const [binSize, setBinSize] = useState('1d');
  const [strategy, setStrategy] = useState('');
  const [strategies, setStrategies] = useState([]);
  const [optType, setOptType] = useState('BACKTESTING');
  const [isRunning, setIsRunning] = useState(false);
  const [runningTid, setRunningTid] = useState(null);
  const [isError, setIsError] = useState(false);
  const [testNameHelperText, setTestNameHelperText] = useState('');
  const [cash, setCash] = useState(10000);
  const [commissions, setCommissions] = useState(0.01);
  const [strategiesParameters, setStrategiesParameters] = useState([]);
  const [paramValues, setParamValues] = useState({});
  const [startDate, setStartDate] = useState(new Date(2000, 1, 1));
  const [endDate, setEndDate] = useState(new Date());
  const [loadingProviders, setLoadingProviders] = useState(true);
  const [loadingSymbols, setLoadingSymbols] = useState(false);
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
        setStrategiesParameters(Object.keys(data));
        setParamValues(data);
      })
      .catch((e) => notify(`Failed to load strategy params: ${e}`, 'error'));
  }, [notify]);

  const updateSymbolsAvailable = useCallback((providerName) => {
    if (!providerName) return;
    setLoadingSymbols(true);
    fetch(`${process.env.REACT_APP_REST_API_URL}/datasource/${providerName}/symbols`)
      .then((response) => response.json())
      .then((data) => {
        setSymbols(data);
        setSymbol(data.length > 0 ? data[0] : '');
        setLoadingSymbols(false);
      })
      .catch((e) => {
        notify(`Failed to load symbols: ${e}`, 'error');
        setLoadingSymbols(false);
      });
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

    setLoadingProviders(true);
    fetch(`${process.env.REACT_APP_REST_API_URL}/datasource/available`)
      .then((response) => response.json())
      .then((data) => {
        setProviders(data);
        if (data.length > 0) {
          setProvider(data[0]);
          updateSymbolsAvailable(data[0]);
        }
        setLoadingProviders(false);
      })
      .catch((e) => {
        notify(`Failed to load data sources: ${e}`, 'error');
        setLoadingProviders(false);
      });
  }, [updateStrategyParams, updateSymbolsAvailable, notify]);

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
    const value = e.target.value;
    setProvider(value);
    setSymbol('');
    updateSymbolsAvailable(value);
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
      strategy_params: {},
    };

    strategiesParameters.forEach((p) => {
      params.strategy_params[p] = paramValues[p];
    });

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
        <FormControl sx={formControlSx}>
          {loadingSymbols ? (
            <Skeleton variant="rectangular" height={56} />
          ) : (
            <>
              <InputLabel id="symbol-select-label">Symbol</InputLabel>
              <Select
                labelId="symbol-select-label"
                id="symbol-select"
                value={symbol}
                label="Symbol"
                disabled={!provider || loadingSymbols}
                onChange={(e) => setSymbol(e.target.value)}
              >
                {symbols.map((p) => (
                  <MenuItem key={p} value={p}>
                    {p}
                  </MenuItem>
                ))}
              </Select>
              {!provider && (
                <FormHelperText>Select a provider first</FormHelperText>
              )}
            </>
          )}
        </FormControl>
        <FormControl sx={formControlSx}>
          <InputLabel id="binsize-select-label">Time Frame</InputLabel>
          <Select
            labelId="binsize-select-label"
            id="binsize-select"
            value={binSize}
            label="Time Frame"
            onChange={(e) => setBinSize(e.target.value)}
          >
            <MenuItem key="1d" value="1d">
              1 Day
            </MenuItem>
            <MenuItem key="1h" value="1h">
              1 Hour
            </MenuItem>
            <MenuItem key="5m" value="5m">
              5 Minutes
            </MenuItem>
            <MenuItem key="1m" value="1m">
              1 Minutes
            </MenuItem>
          </Select>
        </FormControl>
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
        <FormControl sx={formControlSx}>
          {strategiesParameters.map((row) => (
            <FormControl key={row + '-form-control'} sx={formControlSx}>
              <TextField
                id={row + '-text'}
                label={row}
                type="number"
                value={paramValues[row] || ''}
                onChange={(e) => handleParamChange(row, Number(e.target.value))}
              />
            </FormControl>
          ))}
        </FormControl>
        <FormControl sx={formControlSx}>
          <FormLabel>Optimization Type</FormLabel>
          <RadioGroup
            id="optType"
            name="optType"
            value={optType}
            onChange={(e) => setOptType(e.target.value)}
          >
            <FormControlLabel value="BACKTESTING" control={<Radio />} label="Backtesting" />
            <FormControlLabel value="WALKFORWARD" control={<Radio />} label="Walk Forward" />
            <FormControlLabel value="BOTH" control={<Radio />} label="Backtesting + Walk Forward" />
          </RadioGroup>
        </FormControl>
        <Grid container direction="row" justifyContent="flex-end" alignItems="center">
          <Box>
            <IconButton aria-label="Run new test" onClick={submitTest} sx={{ m: 0.5 }}>
              <PlayCircleFilledWhiteIcon color="primary" fontSize="large" />
            </IconButton>
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
