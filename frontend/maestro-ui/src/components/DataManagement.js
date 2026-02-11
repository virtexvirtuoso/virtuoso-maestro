// Data Management page for viewing and downloading OHLCV data from providers
import React, { useState, useEffect, useCallback } from 'react';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Chip from '@mui/material/Chip';
import IconButton from '@mui/material/IconButton';
import Button from '@mui/material/Button';
import Box from '@mui/material/Box';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import TextField from '@mui/material/TextField';
import LinearProgress from '@mui/material/LinearProgress';
import Tooltip from '@mui/material/Tooltip';
import RefreshIcon from '@mui/icons-material/Refresh';
import DownloadIcon from '@mui/icons-material/Download';
import AddIcon from '@mui/icons-material/Add';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import WarningIcon from '@mui/icons-material/Warning';
import Title from './Title';
import EmptyState from './EmptyState';
import { useNotification } from '../context/NotificationContext';

const TIMEFRAMES = [
  { value: '1d', label: '1 Day' },
  { value: '1h', label: '1 Hour' },
  { value: '5m', label: '5 Minutes' },
  { value: '1m', label: '1 Minute' },
];

export default function DataManagement() {
  const [dataSources, setDataSources] = useState([]);
  const [loading, setLoading] = useState(true);
  const [providers, setProviders] = useState([]);
  const [downloadingRows, setDownloadingRows] = useState(new Set());

  // Add dialog state
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [newProvider, setNewProvider] = useState('');
  const [newSymbol, setNewSymbol] = useState('');
  const [newTimeframe, setNewTimeframe] = useState('1d');
  const [addingNew, setAddingNew] = useState(false);

  const { notify } = useNotification();

  // Fetch available providers
  const fetchProviders = useCallback(async () => {
    try {
      const response = await fetch(`${process.env.REACT_APP_REST_API_URL}/datasource/available`);
      const data = await response.json();
      setProviders(data);
      if (data.length > 0 && !newProvider) {
        setNewProvider(data[0]);
      }
    } catch (e) {
      notify(`Failed to load providers: ${e}`, 'error');
    }
  }, [notify, newProvider]);

  // Fetch data sources with metadata
  const fetchDataSources = useCallback(async () => {
    setLoading(true);
    try {
      const providersResponse = await fetch(`${process.env.REACT_APP_REST_API_URL}/datasource/available`);
      const providersList = await providersResponse.json();
      setProviders(providersList);

      const allSources = [];

      for (const provider of providersList) {
        try {
          const symbolsResponse = await fetch(`${process.env.REACT_APP_REST_API_URL}/datasource/${provider}/symbols`);
          const symbols = await symbolsResponse.json();

          for (const symbol of symbols) {
            // Check each timeframe for this provider/symbol combination
            for (const tf of TIMEFRAMES) {
              try {
                // Fetch a small sample to check data availability
                const dataResponse = await fetch(
                  `${process.env.REACT_APP_REST_API_URL}/datasource/${provider}/${symbol}/${tf.value}/`
                );
                const result = await dataResponse.json();

                if (result.data && result.data.length > 0) {
                  // Calculate data stats
                  const records = result.data.length;
                  const sortedData = [...result.data].sort((a, b) =>
                    new Date(a.timestamp?.epoch_time ? a.timestamp.epoch_time * 1000 : a.timestamp) -
                    new Date(b.timestamp?.epoch_time ? b.timestamp.epoch_time * 1000 : b.timestamp)
                  );

                  const lastRecord = sortedData[sortedData.length - 1];

                  const lastUpdate = lastRecord?.timestamp?.epoch_time
                    ? new Date(lastRecord.timestamp.epoch_time * 1000)
                    : new Date(lastRecord?.timestamp);

                  // Check for gaps (simple heuristic: more than expected time between records)
                  const hasGaps = detectGaps(sortedData, tf.value);

                  allSources.push({
                    id: `${provider}-${symbol}-${tf.value}`,
                    provider: provider,
                    symbol: symbol.toUpperCase(),
                    timeframe: tf.value,
                    timeframeLabel: tf.label,
                    records: records,
                    lastUpdate: lastUpdate,
                    status: hasGaps ? 'gaps' : 'ok',
                  });
                }
              } catch (e) {
                // Timeframe not available for this symbol, skip
              }
            }
          }
        } catch (e) {
          // Provider symbols fetch failed, skip
        }
      }

      setDataSources(allSources);
    } catch (e) {
      notify(`Failed to load data sources: ${e}`, 'error');
    } finally {
      setLoading(false);
    }
  }, [notify]);

  // Detect gaps in time series data
  const detectGaps = (data, timeframe) => {
    if (data.length < 2) return false;

    const expectedIntervalMs = {
      '1m': 60 * 1000,
      '5m': 5 * 60 * 1000,
      '1h': 60 * 60 * 1000,
      '1d': 24 * 60 * 60 * 1000,
    }[timeframe];

    if (!expectedIntervalMs) return false;

    // Check a sample of data points for gaps (don't check all for performance)
    const sampleSize = Math.min(100, data.length);
    const step = Math.floor(data.length / sampleSize);

    for (let i = step; i < data.length; i += step) {
      const prevTs = data[i - step].timestamp?.epoch_time
        ? data[i - step].timestamp.epoch_time * 1000
        : new Date(data[i - step].timestamp).getTime();
      const currTs = data[i].timestamp?.epoch_time
        ? data[i].timestamp.epoch_time * 1000
        : new Date(data[i].timestamp).getTime();

      const diff = currTs - prevTs;
      // Allow for 2x expected interval as tolerance
      if (diff > expectedIntervalMs * step * 2) {
        return true;
      }
    }

    return false;
  };

  useEffect(() => {
    fetchDataSources();
    fetchProviders();
  }, [fetchDataSources, fetchProviders]);

  // Handle download for a specific data source
  const handleDownload = async (source) => {
    const rowId = source.id;
    setDownloadingRows(prev => new Set([...prev, rowId]));

    try {
      // Trigger data refresh/download via API
      // Note: This would typically call a backend endpoint to refresh data
      // For now, we'll simulate the download by re-fetching the data
      await fetch(
        `${process.env.REACT_APP_REST_API_URL}/datasource/${source.provider}/${source.symbol}/${source.timeframe}/`
      );

      // Simulate download delay
      await new Promise(resolve => setTimeout(resolve, 1500));

      notify(`Data refreshed for ${source.symbol} (${source.timeframeLabel})`, 'success');

      // Refresh the data sources list
      await fetchDataSources();
    } catch (e) {
      notify(`Download failed: ${e}`, 'error');
    } finally {
      setDownloadingRows(prev => {
        const next = new Set(prev);
        next.delete(rowId);
        return next;
      });
    }
  };

  // Handle adding a new data source
  const handleAddNew = async () => {
    if (!newProvider || !newSymbol || !newTimeframe) {
      notify('Please fill in all fields', 'warning');
      return;
    }

    setAddingNew(true);

    try {
      // Try to fetch data for the new symbol
      const response = await fetch(
        `${process.env.REACT_APP_REST_API_URL}/datasource/${newProvider}/${newSymbol.toUpperCase()}/${newTimeframe}/`
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch data for ${newSymbol}`);
      }

      notify(`Added ${newSymbol.toUpperCase()} (${newTimeframe}) from ${newProvider}`, 'success');
      setAddDialogOpen(false);
      setNewSymbol('');

      // Refresh the data sources list
      await fetchDataSources();
    } catch (e) {
      notify(`Failed to add data source: ${e.message}`, 'error');
    } finally {
      setAddingNew(false);
    }
  };

  const formatDate = (date) => {
    if (!date || isNaN(date.getTime())) return 'N/A';
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const formatNumber = (num) => {
    if (num === undefined || num === null) return 'N/A';
    return num.toLocaleString();
  };

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Title>Data Sources</Title>
            <Box>
              <Tooltip title="Add New Symbol">
                <IconButton
                  color="primary"
                  onClick={() => setAddDialogOpen(true)}
                  sx={{ mr: 1 }}
                >
                  <AddIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title="Refresh All">
                <IconButton
                  color="primary"
                  onClick={fetchDataSources}
                  disabled={loading}
                >
                  <RefreshIcon />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>

          {loading && <LinearProgress sx={{ mb: 2 }} />}

          {!loading && dataSources.length === 0 ? (
            <EmptyState
              title="No Data Sources"
              description="No OHLCV data found. Use the Add button to download data for a new symbol."
              actionLabel="Add Data Source"
              onAction={() => setAddDialogOpen(true)}
            />
          ) : (
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 'bold' }}>Provider</TableCell>
                    <TableCell sx={{ fontWeight: 'bold' }}>Symbol</TableCell>
                    <TableCell sx={{ fontWeight: 'bold' }}>Timeframe</TableCell>
                    <TableCell sx={{ fontWeight: 'bold' }} align="right">Records</TableCell>
                    <TableCell sx={{ fontWeight: 'bold' }}>Last Update</TableCell>
                    <TableCell sx={{ fontWeight: 'bold' }}>Status</TableCell>
                    <TableCell sx={{ fontWeight: 'bold' }} align="center">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {dataSources.map((source) => {
                    const isDownloading = downloadingRows.has(source.id);
                    return (
                      <TableRow key={source.id} hover>
                        <TableCell>
                          <Chip
                            label={source.provider}
                            size="small"
                            variant="outlined"
                            color="primary"
                          />
                        </TableCell>
                        <TableCell sx={{ fontWeight: 500 }}>{source.symbol}</TableCell>
                        <TableCell>{source.timeframeLabel}</TableCell>
                        <TableCell align="right">{formatNumber(source.records)}</TableCell>
                        <TableCell>{formatDate(source.lastUpdate)}</TableCell>
                        <TableCell>
                          {source.status === 'ok' ? (
                            <Chip
                              icon={<CheckCircleIcon />}
                              label="OK"
                              size="small"
                              color="success"
                              variant="filled"
                            />
                          ) : (
                            <Chip
                              icon={<WarningIcon />}
                              label="Gaps"
                              size="small"
                              color="warning"
                              variant="filled"
                            />
                          )}
                        </TableCell>
                        <TableCell align="center">
                          {isDownloading ? (
                            <Box sx={{ width: 40, display: 'inline-block' }}>
                              <LinearProgress />
                            </Box>
                          ) : (
                            <Tooltip title="Download/Refresh Data">
                              <IconButton
                                size="small"
                                color="primary"
                                onClick={() => handleDownload(source)}
                              >
                                <DownloadIcon />
                              </IconButton>
                            </Tooltip>
                          )}
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </Paper>
      </Grid>

      {/* Add New Symbol Dialog */}
      <Dialog open={addDialogOpen} onClose={() => setAddDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add New Data Source</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <FormControl fullWidth>
              <InputLabel id="add-provider-label">Provider</InputLabel>
              <Select
                labelId="add-provider-label"
                value={newProvider}
                label="Provider"
                onChange={(e) => setNewProvider(e.target.value)}
              >
                {providers.map((p) => (
                  <MenuItem key={p} value={p}>{p}</MenuItem>
                ))}
              </Select>
            </FormControl>

            <TextField
              fullWidth
              label="Symbol"
              value={newSymbol}
              onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
              placeholder="e.g., BTCUSDT"
              helperText="Enter the trading pair symbol"
            />

            <FormControl fullWidth>
              <InputLabel id="add-timeframe-label">Timeframe</InputLabel>
              <Select
                labelId="add-timeframe-label"
                value={newTimeframe}
                label="Timeframe"
                onChange={(e) => setNewTimeframe(e.target.value)}
              >
                {TIMEFRAMES.map((tf) => (
                  <MenuItem key={tf.value} value={tf.value}>{tf.label}</MenuItem>
                ))}
              </Select>
            </FormControl>

            {addingNew && <LinearProgress />}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAddDialogOpen(false)} disabled={addingNew}>
            Cancel
          </Button>
          <Button
            onClick={handleAddNew}
            variant="contained"
            disabled={addingNew || !newSymbol}
            startIcon={<DownloadIcon />}
          >
            Download
          </Button>
        </DialogActions>
      </Dialog>
    </Grid>
  );
}
