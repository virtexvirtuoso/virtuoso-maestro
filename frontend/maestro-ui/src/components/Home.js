// Home - Data-aware dashboard with provider cards, stats, and strategy sidebar
import React, { useState, useEffect, useCallback } from 'react';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardActionArea from '@mui/material/CardActionArea';
import Chip from '@mui/material/Chip';
import Button from '@mui/material/Button';
import Skeleton from '@mui/material/Skeleton';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import Divider from '@mui/material/Divider';
import AddIcon from '@mui/icons-material/Add';
import StorageIcon from '@mui/icons-material/Storage';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import HistoryIcon from '@mui/icons-material/History';
import CandleStickChart from './CandleStickChart';
import OptimizationWizard from './OptimizationWizard';
import { useNotification } from '../context/NotificationContext';

// Provider metadata with colors and descriptions
const PROVIDER_CONFIG = {
  binance: { name: 'Binance', color: '#F0B90B', description: 'Largest crypto exchange' },
  bitmex: { name: 'BitMEX', color: '#5B98D8', description: 'Perpetual contracts pioneer' },
  bybit: { name: 'Bybit', color: '#F7A600', description: 'Derivatives trading' },
  mexc: { name: 'MEXC', color: '#2CA6A4', description: 'Wide altcoin selection' },
  kucoin: { name: 'KuCoin', color: '#23AF91', description: 'Major altcoin exchange' },
  gate: { name: 'Gate.io', color: '#17E6A1', description: 'Spot and margin' },
};

// Strategy categories
const STRATEGY_CATEGORIES = [
  { key: 'technical', name: 'Technical', count: 19 },
  { key: 'hybrids', name: 'Hybrids', count: 15 },
  { key: 'derivatives', name: 'Derivatives', count: 11 },
  { key: 'scalping', name: 'Scalping', count: 8 },
  { key: 'momentum', name: 'Momentum', count: 6 },
  { key: 'composite', name: 'Composite', count: 6 },
];

// Stats card component
function StatCard({ icon, value, label, loading }) {
  return (
    <Paper
      sx={{
        p: 2,
        display: 'flex',
        alignItems: 'center',
        gap: 2,
        bgcolor: 'background.paper',
      }}
    >
      <Box sx={{ color: 'primary.main' }}>{icon}</Box>
      <Box>
        {loading ? (
          <Skeleton width={40} height={32} />
        ) : (
          <Typography variant="h4" sx={{ fontWeight: 700, fontFamily: '"IBM Plex Mono", monospace' }}>
            {value}
          </Typography>
        )}
        <Typography variant="body2" color="text.secondary">
          {label}
        </Typography>
      </Box>
    </Paper>
  );
}

// Provider card component
function ProviderCard({ provider, symbols, onClick }) {
  const config = PROVIDER_CONFIG[provider] || { name: provider, color: '#666', description: '' };
  const displaySymbols = symbols.slice(0, 3);
  const moreCount = symbols.length - 3;

  return (
    <Card
      sx={{
        height: '100%',
        borderLeft: `4px solid ${config.color}`,
        '&:hover': { boxShadow: 4 },
        transition: 'box-shadow 0.2s',
      }}
    >
      <CardActionArea onClick={() => onClick(provider)} sx={{ height: '100%' }}>
        <CardContent>
          <Typography variant="h6" sx={{ fontWeight: 600, mb: 0.5 }}>
            {config.name}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
            {config.description}
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            <Chip
              label={`${symbols.length} symbols`}
              size="small"
              sx={{ bgcolor: config.color, color: '#000', fontWeight: 500 }}
            />
          </Box>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
            {displaySymbols.map((sym) => (
              <Chip
                key={sym}
                label={sym.toUpperCase()}
                size="small"
                variant="outlined"
                sx={{ fontSize: '0.7rem' }}
              />
            ))}
            {moreCount > 0 && (
              <Chip
                label={`+${moreCount} more`}
                size="small"
                variant="outlined"
                sx={{ fontSize: '0.7rem', fontStyle: 'italic' }}
              />
            )}
          </Box>
        </CardContent>
      </CardActionArea>
    </Card>
  );
}

export default function Home() {
  const [provider, setProvider] = useState('');
  const [symbol, setSymbol] = useState('');
  const [providers, setProviders] = useState([]);
  const [providerSymbols, setProviderSymbols] = useState({});
  const [strategies, setStrategies] = useState([]);
  const [recentOptimizations, setRecentOptimizations] = useState([]);
  const [loadingProviders, setLoadingProviders] = useState(true);
  const [wizardOpen, setWizardOpen] = useState(false);

  const { notify } = useNotification();

  // Fetch all provider symbols for dashboard display
  const fetchAllProviderSymbols = useCallback(async (providerList) => {
    const symbolMap = {};
    for (const prov of providerList) {
      try {
        const response = await fetch(`${process.env.REACT_APP_REST_API_URL}/datasource/${prov}/symbols`);
        const data = await response.json();
        symbolMap[prov] = data;
      } catch {
        symbolMap[prov] = [];
      }
    }
    setProviderSymbols(symbolMap);
  }, []);

  // Update symbol when provider changes (for chart preview)
  const updateSymbolForChart = useCallback((providerName) => {
    if (!providerName) {
      setSymbol('');
      return;
    }
    fetch(`${process.env.REACT_APP_REST_API_URL}/datasource/${providerName}/symbols`)
      .then((response) => response.json())
      .then((data) => {
        setSymbol(data.length > 0 ? data[0] : '');
      })
      .catch((e) => {
        notify(`Failed to load symbols: ${e}`, 'error');
      });
  }, [notify]);

  // Fetch strategies
  useEffect(() => {
    fetch(`${process.env.REACT_APP_REST_API_URL}/strategy/available`)
      .then((response) => response.json())
      .then((data) => setStrategies(data))
      .catch(() => setStrategies(Array(65).fill('strategy')));
  }, []);

  // Fetch recent optimizations
  useEffect(() => {
    fetch(`${process.env.REACT_APP_REST_API_URL}/optimization/recent?limit=5`)
      .then((response) => {
        if (response.ok) return response.json();
        return [];
      })
      .then((data) => setRecentOptimizations(data))
      .catch(() => setRecentOptimizations([]));
  }, []);

  // Fetch providers
  useEffect(() => {
    setLoadingProviders(true);
    fetch(`${process.env.REACT_APP_REST_API_URL}/datasource/available`)
      .then((response) => response.json())
      .then((data) => {
        setProviders(data);
        fetchAllProviderSymbols(data);
        if (data.length > 0) {
          setProvider(data[0]);
          updateSymbolForChart(data[0]);
        }
        setLoadingProviders(false);
      })
      .catch((e) => {
        notify(`Failed to load data sources: ${e}`, 'error');
        setLoadingProviders(false);
      });
  }, [updateSymbolForChart, fetchAllProviderSymbols, notify]);

  // Handle provider card click - updates chart preview
  const handleProviderCardClick = (selectedProvider) => {
    setProvider(selectedProvider);
    setSymbol('');
    updateSymbolForChart(selectedProvider);
  };

  const openOptimizationWizard = () => {
    setWizardOpen(true);
  };

  const closeOptimizationWizard = () => {
    setWizardOpen(false);
  };

  // Calculate total symbols
  const totalSymbols = Object.values(providerSymbols).reduce(
    (sum, syms) => sum + syms.length,
    0
  );

  return (
    <>
      {/* Stats Row */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={6} sm={3}>
          <StatCard
            icon={<StorageIcon fontSize="large" />}
            value={providers.length || 6}
            label="Providers"
            loading={loadingProviders}
          />
        </Grid>
        <Grid item xs={6} sm={3}>
          <StatCard
            icon={<ShowChartIcon fontSize="large" />}
            value={totalSymbols || 31}
            label="Symbols"
            loading={loadingProviders}
          />
        </Grid>
        <Grid item xs={6} sm={3}>
          <StatCard
            icon={<TrendingUpIcon fontSize="large" />}
            value={strategies.length || 65}
            label="Strategies"
            loading={false}
          />
        </Grid>
        <Grid item xs={6} sm={3}>
          <StatCard
            icon={<HistoryIcon fontSize="large" />}
            value={recentOptimizations.length}
            label="Recent Runs"
            loading={false}
          />
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Main Content */}
        <Grid item xs={12} md={9}>
          {/* New Optimization CTA */}
          <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="h5" sx={{ fontWeight: 600 }}>
              Data Sources
            </Typography>
            <Button
              variant="contained"
              color="primary"
              startIcon={<AddIcon />}
              onClick={openOptimizationWizard}
              sx={{ fontWeight: 600 }}
            >
              New Optimization
            </Button>
          </Box>

          {/* Provider Cards Grid */}
          <Grid container spacing={2} sx={{ mb: 3 }}>
            {loadingProviders ? (
              // Skeleton loading state
              [1, 2, 3, 4, 5, 6].map((i) => (
                <Grid item xs={12} sm={6} md={4} key={i}>
                  <Skeleton variant="rectangular" height={140} sx={{ borderRadius: 1 }} />
                </Grid>
              ))
            ) : (
              providers.map((prov) => (
                <Grid item xs={12} sm={6} md={4} key={prov}>
                  <ProviderCard
                    provider={prov}
                    symbols={providerSymbols[prov] || []}
                    onClick={handleProviderCardClick}
                  />
                </Grid>
              ))
            )}
          </Grid>

          {/* Chart */}
          <Paper sx={{ p: 2, mb: 3 }}>
            <CandleStickChart
              provider={provider}
              symbol={symbol}
              timeframe="1d"
            />
          </Paper>

          {/* Recent Optimizations */}
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
              Recent Optimizations
            </Typography>
            {recentOptimizations.length === 0 ? (
              <Box
                sx={{
                  py: 4,
                  textAlign: 'center',
                  color: 'text.secondary',
                }}
              >
                <HistoryIcon sx={{ fontSize: 48, mb: 1, opacity: 0.5 }} />
                <Typography>No recent optimizations</Typography>
                <Typography variant="body2" sx={{ mt: 1 }}>
                  Click "New Optimization" to run your first backtest
                </Typography>
              </Box>
            ) : (
              <List dense>
                {recentOptimizations.map((opt, idx) => (
                  <React.Fragment key={opt.tid || idx}>
                    <ListItem>
                      <ListItemText
                        primary={opt.test_name || 'Unnamed Test'}
                        secondary={`${opt.strategy} · ${opt.symbol} · ${opt.kind}`}
                      />
                      <Chip
                        label={opt.status || 'completed'}
                        size="small"
                        color={opt.status === 'completed' ? 'success' : 'default'}
                      />
                    </ListItem>
                    {idx < recentOptimizations.length - 1 && <Divider />}
                  </React.Fragment>
                ))}
              </List>
            )}
          </Paper>
        </Grid>

        {/* Sidebar - Strategy Categories */}
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2, position: 'sticky', top: 80 }}>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
              Strategy Categories
            </Typography>
            <List dense>
              {STRATEGY_CATEGORIES.map((cat) => (
                <ListItem
                  key={cat.key}
                  sx={{
                    borderRadius: 1,
                    mb: 0.5,
                    '&:hover': { bgcolor: 'action.hover' },
                  }}
                >
                  <ListItemText
                    primary={cat.name}
                    primaryTypographyProps={{ fontWeight: 500 }}
                  />
                  <Chip
                    label={cat.count}
                    size="small"
                    sx={{ minWidth: 32 }}
                  />
                </ListItem>
              ))}
            </List>
            <Divider sx={{ my: 2 }} />
            <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center' }}>
              {strategies.length || 65} total strategies
            </Typography>
          </Paper>
        </Grid>
      </Grid>

      {/* Optimization Wizard */}
      <OptimizationWizard open={wizardOpen} onClose={closeOptimizationWizard} />
    </>
  );
}
