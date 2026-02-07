// OptimizationWizard - 4-step wizard modal for new optimizations
// Uses progressive disclosure to reduce cognitive load
import React, { useState, useEffect, useCallback } from 'react';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import Box from '@mui/material/Box';
import Stepper from '@mui/material/Stepper';
import Step from '@mui/material/Step';
import StepLabel from '@mui/material/StepLabel';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardActionArea from '@mui/material/CardActionArea';
import Chip from '@mui/material/Chip';
import TextField from '@mui/material/TextField';
import InputAdornment from '@mui/material/InputAdornment';
import Tooltip from '@mui/material/Tooltip';
import Skeleton from '@mui/material/Skeleton';
import Divider from '@mui/material/Divider';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import CloseIcon from '@mui/icons-material/Close';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import { DateTimePicker } from '@mui/x-date-pickers/DateTimePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { useNotification } from '../context/NotificationContext';
import StrategyCategorySelector from './StrategyCategorySelector';
import StrategyParamEditor from './StrategyParamEditor';
import TimeframeSelector from './TimeframeSelector';
import OptimizationTypeSelector from './OptimizationTypeSelector';

// Step labels
const STEPS = ['Data Source', 'Strategy', 'Settings', 'Review'];

// Provider metadata with colors and descriptions
const PROVIDER_CONFIG = {
  binance: { name: 'Binance', color: '#F0B90B', description: 'Largest crypto exchange' },
  bitmex: { name: 'BitMEX', color: '#5B98D8', description: 'Perpetual contracts pioneer' },
  bybit: { name: 'Bybit', color: '#F7A600', description: 'Derivatives trading' },
  mexc: { name: 'MEXC', color: '#2CA6A4', description: 'Wide altcoin selection' },
  kucoin: { name: 'KuCoin', color: '#23AF91', description: 'Major altcoin exchange' },
  gate: { name: 'Gate.io', color: '#17E6A1', description: 'Spot and margin' },
};

// Timeframe options (used for display in Review step)
const TIMEFRAMES = [
  { value: '1m', label: '1 Min' },
  { value: '5m', label: '5 Min' },
  { value: '15m', label: '15 Min' },
  { value: '1h', label: '1 Hour' },
  { value: '4h', label: '4 Hour' },
  { value: '1d', label: '1 Day' },
];


// Optimization type labels for review display
const OPTIMIZATION_TYPE_LABELS = {
  BACKTESTING: 'Backtest',
  WALKFORWARD: 'Walk-Forward',
  BOTH: 'Both',
};

export default function OptimizationWizard({ open, onClose }) {
  const { notify } = useNotification();

  // Wizard state
  const [currentStep, setCurrentStep] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Form data
  const [formData, setFormData] = useState({
    // Step 1: Data Source
    provider: '',
    symbol: '',
    binSize: '1d',
    startDate: new Date(2020, 0, 1),
    endDate: new Date(),
    // Step 2: Strategy
    strategy: '',
    strategyParams: {},
    // Step 3: Settings
    optType: 'BACKTESTING',
    cash: 10000,
    commissions: 0.1,
    wfoSplits: 10,
    // Step 4: Review
    testName: '',
  });

  // Data loading states
  const [providers, setProviders] = useState([]);
  const [symbols, setSymbols] = useState([]);
  const [strategies, setStrategies] = useState([]);
  const [loadingProviders, setLoadingProviders] = useState(true);
  const [loadingSymbols, setLoadingSymbols] = useState(false);
  const [loadingStrategies, setLoadingStrategies] = useState(true);


  // Fetch providers on mount
  useEffect(() => {
    if (!open) return;

    setLoadingProviders(true);
    fetch(`${process.env.REACT_APP_REST_API_URL}/datasource/available`)
      .then((res) => res.json())
      .then((data) => {
        setProviders(data);
        setLoadingProviders(false);
      })
      .catch((e) => {
        notify(`Failed to load providers: ${e}`, 'error');
        setLoadingProviders(false);
      });
  }, [open, notify]);

  // Fetch strategies on mount
  useEffect(() => {
    if (!open) return;

    setLoadingStrategies(true);
    fetch(`${process.env.REACT_APP_REST_API_URL}/strategy/available`)
      .then((res) => res.json())
      .then((data) => {
        setStrategies(data);
        setLoadingStrategies(false);
      })
      .catch((e) => {
        notify(`Failed to load strategies: ${e}`, 'error');
        setLoadingStrategies(false);
      });
  }, [open, notify]);

  // Fetch symbols when provider changes
  const fetchSymbols = useCallback((provider) => {
    if (!provider) {
      setSymbols([]);
      return;
    }
    setLoadingSymbols(true);
    fetch(`${process.env.REACT_APP_REST_API_URL}/datasource/${provider}/symbols`)
      .then((res) => res.json())
      .then((data) => {
        setSymbols(data);
        setLoadingSymbols(false);
      })
      .catch((e) => {
        notify(`Failed to load symbols: ${e}`, 'error');
        setLoadingSymbols(false);
      });
  }, [notify]);

  // Fetch strategy params when strategy changes (for initial values)
  const fetchStrategyParams = useCallback((strategy) => {
    if (!strategy) {
      return;
    }
    fetch(`${process.env.REACT_APP_REST_API_URL}/strategy/${strategy}/params`)
      .then((res) => res.json())
      .then((data) => {
        setFormData((prev) => ({
          ...prev,
          strategyParams: data,
        }));
      })
      .catch((e) => {
        notify(`Failed to load strategy params: ${e}`, 'error');
      });
  }, [notify]);

  // Update form data
  const updateFormData = (field, value) => {
    setFormData((prev) => ({ ...prev, [field]: value }));

    // Side effects
    if (field === 'provider') {
      setFormData((prev) => ({ ...prev, symbol: '' }));
      fetchSymbols(value);
    }
    if (field === 'strategy') {
      fetchStrategyParams(value);
    }
  };

  // Update strategy param
  const updateStrategyParam = (param, value) => {
    setFormData((prev) => ({
      ...prev,
      strategyParams: {
        ...prev.strategyParams,
        [param]: value,
      },
    }));
  };

  // Validation per step
  const isStepValid = (step) => {
    switch (step) {
      case 0: // Data Source
        return formData.provider && formData.symbol && formData.binSize;
      case 1: // Strategy
        return formData.strategy;
      case 2: // Settings
        return formData.optType && formData.cash > 0;
      case 3: // Review
        return formData.testName.trim().length > 0;
      default:
        return true;
    }
  };

  // Navigation handlers
  const handleNext = () => {
    if (currentStep < STEPS.length - 1) {
      setCurrentStep((prev) => prev + 1);
    }
  };

  const handleBack = () => {
    if (currentStep > 0) {
      setCurrentStep((prev) => prev - 1);
    }
  };

  // Submit handler
  const handleSubmit = async () => {
    setIsSubmitting(true);

    const payload = {
      test_name: formData.testName,
      symbol: formData.symbol,
      provider: formData.provider,
      bin_size: formData.binSize,
      strategy: formData.strategy,
      kind: formData.optType,
      cash: formData.cash,
      commissions: formData.commissions,
      start_date: formData.startDate.getTime(),
      end_date: formData.endDate.getTime(),
      strategy_params: formData.strategyParams,
      wfo_splits: formData.wfoSplits,
    };

    try {
      const response = await fetch(
        `${process.env.REACT_APP_REST_API_URL}/optimization/new/`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        }
      );

      const data = await response.json();

      if (response.status !== 200) {
        notify(`Error: ${data.error || 'Failed to start optimization'}`, 'error');
      } else {
        notify(`Optimization started: ${data.tid}`, 'success');
        handleClose();
      }
    } catch (e) {
      notify(`Failed to start optimization: ${e}`, 'error');
    } finally {
      setIsSubmitting(false);
    }
  };

  // Close and reset
  const handleClose = () => {
    setCurrentStep(0);
    setFormData({
      provider: '',
      symbol: '',
      binSize: '1d',
      startDate: new Date(2020, 0, 1),
      endDate: new Date(),
      strategy: '',
      strategyParams: {},
      optType: 'BACKTESTING',
      cash: 10000,
      commissions: 0.1,
      wfoSplits: 10,
      testName: '',
    });
    setSymbols([]);
    onClose();
  };

  // Handle ESC key
  const handleKeyDown = (e) => {
    if (e.key === 'Escape') {
      handleClose();
    }
  };


  // Render Step 1: Data Source
  const renderDataSourceStep = () => (
    <LocalizationProvider dateAdapter={AdapterDateFns}>
      <Box>
        {/* Provider Selection */}
        <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
          Select Provider
        </Typography>
        <Grid container spacing={2} sx={{ mb: 3 }}>
          {loadingProviders ? (
            [1, 2, 3, 4].map((i) => (
              <Grid item xs={6} sm={3} key={i}>
                <Skeleton variant="rectangular" height={100} sx={{ borderRadius: 1 }} />
              </Grid>
            ))
          ) : (
            providers.map((prov) => {
              const config = PROVIDER_CONFIG[prov] || { name: prov, color: '#666' };
              const isSelected = formData.provider === prov;
              return (
                <Grid item xs={6} sm={3} key={prov}>
                  <Card
                    sx={{
                      borderLeft: `4px solid ${config.color}`,
                      bgcolor: isSelected ? 'action.selected' : 'background.paper',
                      '&:hover': { boxShadow: 3 },
                      transition: 'all 0.2s',
                    }}
                  >
                    <CardActionArea onClick={() => updateFormData('provider', prov)}>
                      <CardContent sx={{ py: 1.5 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          {isSelected && <CheckCircleIcon color="primary" fontSize="small" />}
                          <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                            {config.name}
                          </Typography>
                        </Box>
                      </CardContent>
                    </CardActionArea>
                  </Card>
                </Grid>
              );
            })
          )}
        </Grid>

        {/* Symbol Selection */}
        <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
          Select Symbol
        </Typography>
        <Box
          sx={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: 1,
            mb: 3,
            p: 2,
            border: '1px solid',
            borderColor: 'divider',
            borderRadius: 1,
            minHeight: 56,
            maxHeight: 160,
            overflowY: 'auto',
          }}
        >
          {!formData.provider ? (
            <Typography variant="body2" color="text.secondary">
              Select a provider first
            </Typography>
          ) : loadingSymbols ? (
            [1, 2, 3, 4, 5, 6].map((i) => (
              <Skeleton key={i} variant="rounded" width={72} height={32} sx={{ borderRadius: '16px' }} />
            ))
          ) : symbols.length === 0 ? (
            <Typography variant="body2" color="text.secondary">
              No symbols available
            </Typography>
          ) : (
            symbols.map((sym) => (
              <Chip
                key={sym}
                label={sym.toUpperCase()}
                variant={formData.symbol === sym ? 'filled' : 'outlined'}
                color={formData.symbol === sym ? 'primary' : 'default'}
                onClick={() => updateFormData('symbol', sym)}
                sx={{ fontWeight: formData.symbol === sym ? 600 : 400 }}
              />
            ))
          )}
        </Box>

        {/* Timeframe Selection */}
        <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
          Timeframe
        </Typography>
        <Box sx={{ mb: 3 }}>
          <TimeframeSelector
            value={formData.binSize}
            onChange={(value) => updateFormData('binSize', value)}
            disabled={!formData.provider || !formData.symbol}
          />
        </Box>

        {/* Date Range */}
        <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
          Date Range
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <DateTimePicker
              label="Start Date"
              value={formData.startDate}
              onChange={(d) => updateFormData('startDate', d)}
              ampm={false}
              format="yyyy/MM/dd HH:mm"
              disableFuture
              slotProps={{ textField: { fullWidth: true } }}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <DateTimePicker
              label="End Date"
              value={formData.endDate}
              onChange={(d) => updateFormData('endDate', d)}
              ampm={false}
              format="yyyy/MM/dd HH:mm"
              disableFuture
              slotProps={{ textField: { fullWidth: true } }}
            />
          </Grid>
        </Grid>
      </Box>
    </LocalizationProvider>
  );

  // Render Step 2: Strategy
  const renderStrategyStep = () => (
    <Box>
      {/* Strategy Category Selector */}
      <StrategyCategorySelector
        value={formData.strategy}
        onChange={(strategyName) => updateFormData('strategy', strategyName)}
        loading={loadingStrategies}
        strategies={strategies}
      />

      {/* Strategy Parameters - Using enhanced StrategyParamEditor */}
      {formData.strategy && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
            Strategy Parameters
          </Typography>
          <StrategyParamEditor
            strategy={formData.strategy}
            values={formData.strategyParams}
            onChange={(param, value) => updateStrategyParam(param, value)}
            showSliders={true}
          />
        </Box>
      )}
    </Box>
  );

  // Render Step 3: Settings
  const renderSettingsStep = () => (
    <Box>
      {/* Optimization Type */}
      <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
        Optimization Type
      </Typography>
      <Box sx={{ mb: 3 }}>
        <OptimizationTypeSelector
          value={formData.optType}
          onChange={(value) => updateFormData('optType', value)}
        />
      </Box>

      {/* Cash & Commission */}
      <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
        Portfolio Settings
      </Typography>
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Starting Cash"
            type="number"
            value={formData.cash}
            onChange={(e) => updateFormData('cash', Number(e.target.value))}
            helperText="Initial portfolio value for simulation"
            InputProps={{
              startAdornment: <InputAdornment position="start">$</InputAdornment>,
            }}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <Tooltip title="Commission fee applied per trade. Enter as percentage (e.g., 0.1 = 0.1%)" arrow>
            <TextField
              fullWidth
              label="Commission"
              type="number"
              value={formData.commissions}
              onChange={(e) => updateFormData('commissions', Number(e.target.value))}
              helperText="Typical: 0.05-0.20% for crypto exchanges"
              InputProps={{
                endAdornment: <InputAdornment position="end">%</InputAdornment>,
              }}
            />
          </Tooltip>
        </Grid>
      </Grid>

      {/* Walk-Forward Settings */}
      {(formData.optType === 'WALKFORWARD' || formData.optType === 'BOTH') && (
        <>
          <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
            Walk-Forward Settings
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Number of Splits"
                type="number"
                value={formData.wfoSplits}
                onChange={(e) => updateFormData('wfoSplits', Number(e.target.value))}
                helperText="Rolling time-series validation windows (default: 10)"
                inputProps={{ min: 2, max: 50 }}
              />
            </Grid>
          </Grid>
        </>
      )}
    </Box>
  );

  // Render Step 4: Review
  const renderReviewStep = () => (
    <Box>
      <Alert severity="info" sx={{ mb: 3 }}>
        Review your optimization settings before running
      </Alert>

      {/* Summary Table */}
      <Paper variant="outlined" sx={{ mb: 3 }}>
        <Table size="small">
          <TableBody>
            <TableRow>
              <TableCell sx={{ fontWeight: 600, width: '30%' }}>Provider</TableCell>
              <TableCell>
                {PROVIDER_CONFIG[formData.provider]?.name || formData.provider}
              </TableCell>
            </TableRow>
            <TableRow>
              <TableCell sx={{ fontWeight: 600 }}>Symbol</TableCell>
              <TableCell>{formData.symbol.toUpperCase()}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell sx={{ fontWeight: 600 }}>Timeframe</TableCell>
              <TableCell>
                {TIMEFRAMES.find((t) => t.value === formData.binSize)?.label || formData.binSize}
              </TableCell>
            </TableRow>
            <TableRow>
              <TableCell sx={{ fontWeight: 600 }}>Date Range</TableCell>
              <TableCell>
                {formData.startDate.toLocaleDateString()} - {formData.endDate.toLocaleDateString()}
              </TableCell>
            </TableRow>
            <TableRow>
              <TableCell sx={{ fontWeight: 600 }}>Strategy</TableCell>
              <TableCell>{formData.strategy}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell sx={{ fontWeight: 600 }}>Optimization Type</TableCell>
              <TableCell>
                {OPTIMIZATION_TYPE_LABELS[formData.optType]}
              </TableCell>
            </TableRow>
            <TableRow>
              <TableCell sx={{ fontWeight: 600 }}>Starting Cash</TableCell>
              <TableCell>${formData.cash.toLocaleString()}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell sx={{ fontWeight: 600 }}>Commission</TableCell>
              <TableCell>{formData.commissions}%</TableCell>
            </TableRow>
            {(formData.optType === 'WALKFORWARD' || formData.optType === 'BOTH') && (
              <TableRow>
                <TableCell sx={{ fontWeight: 600 }}>WFO Splits</TableCell>
                <TableCell>{formData.wfoSplits}</TableCell>
              </TableRow>
            )}
            {Object.keys(formData.strategyParams).length > 0 && (
              <TableRow>
                <TableCell sx={{ fontWeight: 600, verticalAlign: 'top' }}>
                  Strategy Params
                </TableCell>
                <TableCell>
                  {Object.entries(formData.strategyParams).map(([k, v]) => (
                    <Chip
                      key={k}
                      label={`${k}: ${v}`}
                      size="small"
                      sx={{ mr: 0.5, mb: 0.5 }}
                    />
                  ))}
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </Paper>

      {/* Test Name Input */}
      <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
        Test Name
      </Typography>
      <TextField
        fullWidth
        label="Enter a name for this optimization"
        value={formData.testName}
        onChange={(e) => updateFormData('testName', e.target.value)}
        placeholder={`${formData.strategy}_${formData.symbol}_${new Date().toISOString().slice(0, 10)}`}
        helperText="A descriptive name helps identify this test in results"
      />
    </Box>
  );

  // Render current step content
  const renderStepContent = () => {
    switch (currentStep) {
      case 0:
        return renderDataSourceStep();
      case 1:
        return renderStrategyStep();
      case 2:
        return renderSettingsStep();
      case 3:
        return renderReviewStep();
      default:
        return null;
    }
  };

  return (
    <Dialog
      open={open}
      onClose={handleClose}
      onKeyDown={handleKeyDown}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: { minHeight: '70vh', maxHeight: '90vh' },
      }}
    >
      <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>
          New Optimization
        </Typography>
        <IconButton onClick={handleClose} size="small" aria-label="close">
          <CloseIcon />
        </IconButton>
      </DialogTitle>

      <Divider />

      {/* Stepper */}
      <Box sx={{ px: 3, py: 2, bgcolor: 'background.default' }}>
        <Stepper activeStep={currentStep}>
          {STEPS.map((label, index) => (
            <Step key={label} completed={index < currentStep}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
      </Box>

      <Divider />

      {/* Content */}
      <DialogContent sx={{ py: 3 }}>
        {renderStepContent()}
      </DialogContent>

      <Divider />

      {/* Navigation */}
      <DialogActions sx={{ px: 3, py: 2, justifyContent: 'space-between' }}>
        <Button
          onClick={handleBack}
          disabled={currentStep === 0}
          startIcon={<ArrowBackIcon />}
        >
          Back
        </Button>

        {currentStep === STEPS.length - 1 ? (
          <Button
            variant="contained"
            color="primary"
            onClick={handleSubmit}
            disabled={!isStepValid(currentStep) || isSubmitting}
            startIcon={
              isSubmitting ? (
                <CircularProgress size={20} color="inherit" />
              ) : (
                <PlayArrowIcon />
              )
            }
          >
            {isSubmitting ? 'Running...' : 'Run Optimization'}
          </Button>
        ) : (
          <Button
            variant="contained"
            onClick={handleNext}
            disabled={!isStepValid(currentStep)}
            endIcon={<ArrowForwardIcon />}
          >
            Next
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
}
