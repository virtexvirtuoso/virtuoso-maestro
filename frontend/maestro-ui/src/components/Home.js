import React, { useState, useEffect, useCallback } from 'react';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import CandleStickChart from './CandleStickChart';
import OptimizationForm from './OptimizationForm';
import { useNotification } from '../context/NotificationContext';

export default function Home() {
  const [provider, setProvider] = useState('');
  const [symbol, setSymbol] = useState('');
  const [binSize, setBinSize] = useState('1d');
  const [providers, setProviders] = useState([]);
  const [symbols, setSymbols] = useState([]);
  const [loadingProviders, setLoadingProviders] = useState(true);
  const [loadingSymbols, setLoadingSymbols] = useState(false);

  const { notify } = useNotification();

  const updateSymbolsAvailable = useCallback((providerName) => {
    if (!providerName) {
      setSymbols([]);
      return;
    }
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
  }, [updateSymbolsAvailable, notify]);

  const handleProviderChange = (newProvider) => {
    setProvider(newProvider);
    setSymbol('');
    updateSymbolsAvailable(newProvider);
  };

  const handleSymbolChange = (newSymbol) => {
    setSymbol(newSymbol);
  };

  const handleBinSizeChange = (newBinSize) => {
    setBinSize(newBinSize);
  };

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Paper
          sx={{
            p: 2,
            display: 'flex',
            overflow: 'auto',
            flexDirection: 'column',
          }}
        >
          <CandleStickChart
            provider={provider}
            symbol={symbol}
            timeframe={binSize}
          />
        </Paper>
      </Grid>
      <Grid item xs={12}>
        <Paper
          sx={{
            p: 2,
            display: 'flex',
            overflow: 'auto',
            flexDirection: 'column',
          }}
        >
          <OptimizationForm
            provider={provider}
            symbol={symbol}
            binSize={binSize}
            providers={providers}
            symbols={symbols}
            loadingProviders={loadingProviders}
            loadingSymbols={loadingSymbols}
            onProviderChange={handleProviderChange}
            onSymbolChange={handleSymbolChange}
            onBinSizeChange={handleBinSizeChange}
          />
        </Paper>
      </Grid>
    </Grid>
  );
}
