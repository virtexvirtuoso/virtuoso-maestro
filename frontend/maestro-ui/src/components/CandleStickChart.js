import React, { useEffect, useState, useCallback } from 'react';
import { useTheme } from '@mui/material/styles';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Skeleton from '@mui/material/Skeleton';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import RefreshIcon from '@mui/icons-material/Refresh';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import Title from './Title';
import Highcharts from 'highcharts/highstock';
import HighchartsReact from 'highcharts-react-official';
import TradingChart, { convertOHLCVData } from './TradingChart';
import { FEATURES } from '../config/featureFlags';

// Empty state component shown when no data source is selected
function EmptyChartState() {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: 400,
        color: 'text.secondary',
      }}
    >
      <ShowChartIcon sx={{ fontSize: 64, mb: 2, opacity: 0.5 }} />
      <Typography variant="h6" gutterBottom>
        No Data Selected
      </Typography>
      <Typography variant="body2" color="text.secondary">
        Select a data source to view chart
      </Typography>
    </Box>
  );
}

// Error state component shown when data fetch fails
function ErrorChartState({ onRetry }) {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: 400,
        color: 'error.main',
      }}
    >
      <ErrorOutlineIcon sx={{ fontSize: 64, mb: 2 }} />
      <Typography variant="h6" gutterBottom>
        Failed to Load Chart Data
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        There was an error fetching the data. Please try again.
      </Typography>
      <Button
        variant="outlined"
        color="primary"
        startIcon={<RefreshIcon />}
        onClick={onRetry}
      >
        Retry
      </Button>
    </Box>
  );
}

// Loading state component
function LoadingChartState() {
  return (
    <Box sx={{ height: 400 }}>
      <Skeleton variant="rectangular" height={400} animation="wave" />
    </Box>
  );
}

export default function CandleStickChart({ provider, symbol, timeframe = '1d' }) {
  const theme = useTheme();
  const [chart, setChart] = useState(null);
  const [ohlcvData, setOhlcvData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);

  const fetchData = useCallback(() => {
    if (!provider || !symbol) return;

    setLoading(true);
    setError(false);

    const url = `${process.env.REACT_APP_REST_API_URL}/datasource/${provider}/${symbol}/${timeframe}`;
    console.log('Fetching chart data from:', url);

    fetch(url)
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data) => {
        const sortedData = data['data'].sort(
          (a, b) => a['timestamp']['epoch_time'] - b['timestamp']['epoch_time']
        );

        if (FEATURES.USE_TRADINGVIEW) {
          setOhlcvData(convertOHLCVData(sortedData));
        } else {
          const ohlcData = sortedData.map((x) => [
            x['timestamp']['epoch_time'] * 1000,
            x['open'],
            x['high'],
            x['low'],
            x['close'],
          ]);

          const chartTitle = `${symbol.toUpperCase()} - ${timeframe.toUpperCase()}`;

          const options = {
            series: [
              {
                type: 'candlestick',
                name: chartTitle,
                color: theme.palette.secondary.light,
                data: ohlcData,
              },
            ],
            rangeSelector: {
              selected: 1,
            },
          };

          setChart(
            <HighchartsReact
              constructorType={'stockChart'}
              highcharts={Highcharts}
              options={options}
            />
          );
        }
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to fetch chart data:', err);
        setError(true);
        setLoading(false);
      });
  }, [provider, symbol, timeframe, theme.palette.secondary.light]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Guard: Show empty state if no provider or symbol selected
  if (!provider || !symbol) {
    return (
      <React.Fragment>
        <Title>Price Chart</Title>
        <EmptyChartState />
      </React.Fragment>
    );
  }

  // Show loading state
  if (loading) {
    return (
      <React.Fragment>
        <Title>{`${symbol.toUpperCase()} - ${timeframe.toUpperCase()}`}</Title>
        <LoadingChartState />
      </React.Fragment>
    );
  }

  // Show error state with retry button
  if (error) {
    return (
      <React.Fragment>
        <Title>{`${symbol.toUpperCase()} - ${timeframe.toUpperCase()}`}</Title>
        <ErrorChartState onRetry={fetchData} />
      </React.Fragment>
    );
  }

  const chartTitle = `${symbol.toUpperCase()} - ${timeframe.toUpperCase()}`;

  // Render TradingView chart if feature flag is enabled
  if (FEATURES.USE_TRADINGVIEW) {
    return (
      <React.Fragment>
        <Title>{chartTitle}</Title>
        {ohlcvData.length > 0 && (
          <TradingChart
            ohlcvData={ohlcvData}
            title=""
            height={400}
            upColor={theme.palette.success.main}
            downColor={theme.palette.error.main}
          />
        )}
      </React.Fragment>
    );
  }

  // Original Highcharts render
  return (
    <React.Fragment>
      <Title>{chartTitle}</Title>
      {chart}
    </React.Fragment>
  );
}
