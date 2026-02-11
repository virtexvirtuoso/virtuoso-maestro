// Welcome empty state for first-time users with quick start guide
import React from 'react';
import PropTypes from 'prop-types';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Paper from '@mui/material/Paper';
import Fade from '@mui/material/Fade';
import Grid from '@mui/material/Grid';
import Chip from '@mui/material/Chip';
import Card from '@mui/material/Card';
import CardActionArea from '@mui/material/CardActionArea';
import CardContent from '@mui/material/CardContent';
import StorageIcon from '@mui/icons-material/Storage';
import TuneIcon from '@mui/icons-material/Tune';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import BoltIcon from '@mui/icons-material/Bolt';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import SpeedIcon from '@mui/icons-material/Speed';

// Preset backtest templates for quick starts
// NOTE: Symbols must match what's available in the database (check /datasource/{provider}/symbols)
// Available symbols: BTC, ETH, SOL, etc. (NOT BTCUSDT - the DB uses short names)
// Strategy names must match backend: EmaCrossStrategy, MACDStrategy, RSIStrategy, BollingerBandsStrategy
export const PRESET_TEMPLATES = [
  {
    id: 'btc-trend',
    name: 'BTC Trend Following',
    description: 'EMA crossover on Bitcoin daily',
    icon: TrendingUpIcon,
    color: '#F7931A',
    config: {
      provider: 'BINANCE',
      symbol: 'BTC',
      binSize: '1d',
      strategy: 'EmaCrossStrategy',
      strategyParams: { fast_period: 12, slow_period: 26 },
      optType: 'BACKTESTING',
    },
  },
  {
    id: 'btc-rsi',
    name: 'BTC RSI Reversal',
    description: 'Mean reversion on Bitcoin 4h',
    icon: BoltIcon,
    color: '#627EEA',
    config: {
      provider: 'BINANCE',
      symbol: 'BTC',
      binSize: '4h',
      strategy: 'RSIStrategy',
      strategyParams: { period: 14, oversold: 30, overbought: 70 },
      optType: 'BACKTESTING',
    },
  },
  {
    id: 'eth-bollinger',
    name: 'ETH Bollinger Breakout',
    description: 'Volatility breakout on Ethereum hourly',
    icon: AutoGraphIcon,
    color: '#627EEA',
    config: {
      provider: 'BINANCE',
      symbol: 'ETH',
      binSize: '1h',
      strategy: 'BollingerBandsStrategy',
      strategyParams: { period: 20, std_dev: 2.0 },
      optType: 'BACKTESTING',
    },
  },
  {
    id: 'quick-backtest',
    name: 'Quick MACD Test',
    description: 'Fast backtest on BTC daily',
    icon: SpeedIcon,
    color: '#00C853',
    config: {
      provider: 'BINANCE',
      symbol: 'BTC',
      binSize: '1d',
      strategy: 'MACDStrategy',
      strategyParams: { fast_period: 12, slow_period: 26, signal_period: 9 },
      optType: 'BACKTESTING',
    },
  },
];

// Quick start steps for new users
const QUICK_START_STEPS = [
  {
    number: 1,
    title: 'Choose Data Source',
    description: 'Select an exchange and trading pair for backtesting',
    icon: StorageIcon,
  },
  {
    number: 2,
    title: 'Select Strategy',
    description: 'Pick from 65+ technical and hybrid strategies',
    icon: TuneIcon,
  },
  {
    number: 3,
    title: 'Configure & Run',
    description: 'Set parameters and run walk-forward optimization',
    icon: PlayArrowIcon,
  },
];

function StepCard({ step }) {
  const IconComponent = step.icon;
  return (
    <Paper
      elevation={0}
      sx={{
        p: 3,
        textAlign: 'center',
        bgcolor: 'background.paper',
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 2,
        flex: 1,
        minWidth: 200,
        transition: 'all 0.2s ease-in-out',
        '&:hover': {
          borderColor: 'primary.main',
          transform: 'translateY(-2px)',
          boxShadow: '0 4px 20px rgba(251, 191, 36, 0.15)',
        },
      }}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: 48,
          height: 48,
          borderRadius: '50%',
          bgcolor: 'rgba(251, 191, 36, 0.1)',
          border: '2px solid',
          borderColor: 'primary.main',
          mx: 'auto',
          mb: 2,
        }}
      >
        <Typography
          variant="h6"
          sx={{
            fontWeight: 700,
            fontFamily: '"IBM Plex Mono", monospace',
            color: 'primary.main',
          }}
        >
          {step.number}
        </Typography>
      </Box>
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 1,
          mb: 1,
        }}
      >
        <IconComponent sx={{ fontSize: 20, color: 'text.secondary' }} />
        <Typography
          variant="subtitle1"
          sx={{
            fontWeight: 600,
            fontFamily: '"IBM Plex Mono", monospace',
          }}
        >
          {step.title}
        </Typography>
      </Box>
      <Typography variant="body2" color="text.secondary">
        {step.description}
      </Typography>
    </Paper>
  );
}

StepCard.propTypes = {
  step: PropTypes.shape({
    number: PropTypes.number.isRequired,
    title: PropTypes.string.isRequired,
    description: PropTypes.string.isRequired,
    icon: PropTypes.elementType.isRequired,
  }).isRequired,
};

// Preset template card component
function PresetCard({ preset, onClick }) {
  const IconComponent = preset.icon;
  return (
    <Card
      sx={{
        height: '100%',
        borderLeft: `4px solid ${preset.color}`,
        bgcolor: 'background.paper',
        transition: 'all 0.2s ease-in-out',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: `0 8px 24px ${preset.color}33`,
        },
      }}
    >
      <CardActionArea onClick={() => onClick(preset)} sx={{ height: '100%' }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 1 }}>
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: 36,
                height: 36,
                borderRadius: '8px',
                bgcolor: `${preset.color}22`,
              }}
            >
              <IconComponent sx={{ color: preset.color, fontSize: 20 }} />
            </Box>
            <Typography
              variant="subtitle1"
              sx={{
                fontWeight: 600,
                fontFamily: '"IBM Plex Mono", monospace',
              }}
            >
              {preset.name}
            </Typography>
          </Box>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
            {preset.description}
          </Typography>
          <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
            <Chip
              label={preset.config.symbol}
              size="small"
              sx={{ bgcolor: `${preset.color}22`, color: preset.color, fontWeight: 600 }}
            />
            <Chip label={preset.config.binSize} size="small" variant="outlined" />
            <Chip label={preset.config.strategy} size="small" variant="outlined" />
          </Box>
        </CardContent>
      </CardActionArea>
    </Card>
  );
}

PresetCard.propTypes = {
  preset: PropTypes.shape({
    id: PropTypes.string.isRequired,
    name: PropTypes.string.isRequired,
    description: PropTypes.string.isRequired,
    icon: PropTypes.elementType.isRequired,
    color: PropTypes.string.isRequired,
    config: PropTypes.object.isRequired,
  }).isRequired,
  onClick: PropTypes.func.isRequired,
};

function WelcomeEmptyState({ onStartOptimization, onSelectPreset }) {
  return (
    <Fade in timeout={600}>
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '70vh',
          textAlign: 'center',
          px: 3,
        }}
      >
        {/* Logo/Icon */}
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 100,
            height: 100,
            borderRadius: '50%',
            background: 'linear-gradient(135deg, rgba(251, 191, 36, 0.2) 0%, rgba(251, 191, 36, 0.05) 100%)',
            border: '2px solid',
            borderColor: 'primary.main',
            mb: 4,
            boxShadow: '0 0 30px rgba(251, 191, 36, 0.2)',
          }}
        >
          <ShowChartIcon
            sx={{
              fontSize: 50,
              color: 'primary.main',
            }}
          />
        </Box>

        {/* Welcome Message */}
        <Typography
          variant="h4"
          sx={{
            fontWeight: 700,
            fontFamily: '"IBM Plex Mono", monospace',
            color: 'primary.main',
            mb: 1,
            textShadow: '0 0 20px rgba(251, 191, 36, 0.3)',
          }}
        >
          Welcome to Maestro
        </Typography>
        <Typography
          variant="h6"
          sx={{
            fontWeight: 400,
            color: 'text.secondary',
            mb: 4,
            maxWidth: 500,
          }}
        >
          The Master Conductor of Trading Strategies
        </Typography>

        {/* Quick Start Steps */}
        <Typography
          variant="overline"
          sx={{
            color: 'text.secondary',
            letterSpacing: 2,
            mb: 2,
          }}
        >
          Quick Start Guide
        </Typography>
        <Box
          sx={{
            display: 'flex',
            flexDirection: { xs: 'column', md: 'row' },
            gap: 2,
            mb: 4,
            maxWidth: 800,
            width: '100%',
          }}
        >
          {QUICK_START_STEPS.map((step) => (
            <StepCard key={step.number} step={step} />
          ))}
        </Box>

        {/* CTA Button */}
        <Button
          variant="contained"
          color="primary"
          size="large"
          onClick={onStartOptimization}
          startIcon={<PlayArrowIcon />}
          sx={{
            px: 4,
            py: 1.5,
            fontWeight: 700,
            fontFamily: '"IBM Plex Mono", monospace',
            fontSize: '1rem',
            boxShadow: '0 4px 20px rgba(251, 191, 36, 0.3)',
            '&:hover': {
              boxShadow: '0 6px 30px rgba(251, 191, 36, 0.4)',
            },
          }}
        >
          Run First Optimization
        </Button>

        {/* Quick Start Presets */}
        <Box sx={{ mt: 6, width: '100%', maxWidth: 900 }}>
          <Typography
            variant="overline"
            sx={{
              color: 'text.secondary',
              letterSpacing: 2,
              mb: 2,
              display: 'block',
            }}
          >
            Or Choose a Quick Start Template
          </Typography>
          <Grid container spacing={2}>
            {PRESET_TEMPLATES.map((preset) => (
              <Grid item xs={12} sm={6} md={3} key={preset.id}>
                <PresetCard preset={preset} onClick={onSelectPreset} />
              </Grid>
            ))}
          </Grid>
        </Box>

        {/* Subtle hint */}
        <Typography
          variant="caption"
          sx={{
            color: 'text.disabled',
            mt: 4,
            fontStyle: 'italic',
          }}
        >
          Walk-forward optimization prevents overfitting via rolling time-series validation
        </Typography>
      </Box>
    </Fade>
  );
}

WelcomeEmptyState.propTypes = {
  onStartOptimization: PropTypes.func.isRequired,
  onSelectPreset: PropTypes.func.isRequired,
};

export default WelcomeEmptyState;
