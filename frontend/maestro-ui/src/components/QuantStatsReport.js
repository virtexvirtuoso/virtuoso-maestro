// QuantStats report component displaying Returns, Risk, and Ratios metrics
import React from 'react';
import PropTypes from 'prop-types';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Skeleton from '@mui/material/Skeleton';
import Divider from '@mui/material/Divider';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import WarningIcon from '@mui/icons-material/Warning';
import BalanceIcon from '@mui/icons-material/Balance';

const paperSx = {
  p: 2,
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
};

const MetricRow = ({ label, value, isPositive }) => (
  <Box
    sx={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      py: 0.75,
      borderBottom: '1px solid',
      borderColor: 'divider',
      '&:last-child': { borderBottom: 'none' },
    }}
  >
    <Typography
      variant="body2"
      sx={{ color: 'text.secondary', fontFamily: '"IBM Plex Mono", monospace' }}
    >
      {label}
    </Typography>
    <Typography
      variant="body2"
      sx={{
        fontFamily: '"IBM Plex Mono", monospace',
        fontWeight: 600,
        color: isPositive === undefined ? 'text.primary' : isPositive ? 'success.main' : 'error.main',
      }}
    >
      {value}
    </Typography>
  </Box>
);

MetricRow.propTypes = {
  label: PropTypes.string.isRequired,
  value: PropTypes.string.isRequired,
  isPositive: PropTypes.bool,
};

const SectionHeader = ({ icon: Icon, title }) => (
  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
    <Icon sx={{ color: 'primary.main', fontSize: 24 }} />
    <Typography
      variant="subtitle1"
      sx={{ fontFamily: '"IBM Plex Mono", monospace', fontWeight: 600 }}
    >
      {title}
    </Typography>
  </Box>
);

SectionHeader.propTypes = {
  icon: PropTypes.elementType.isRequired,
  title: PropTypes.string.isRequired,
};

const formatPercent = (value, decimals = 2) => {
  if (value === null || value === undefined || isNaN(value)) return 'N/A';
  return `${(value * 100).toFixed(decimals)}%`;
};

const formatDecimal = (value, decimals = 3) => {
  if (value === null || value === undefined || isNaN(value)) return 'N/A';
  return value.toFixed(decimals);
};

const LoadingSkeleton = () => (
  <Grid container spacing={2}>
    {[0, 1, 2].map((i) => (
      <Grid item xs={12} md={4} key={i}>
        <Paper sx={paperSx}>
          <Skeleton variant="text" width={120} height={32} />
          <Divider sx={{ my: 1 }} />
          {[0, 1, 2, 3].map((j) => (
            <Box key={j} sx={{ display: 'flex', justifyContent: 'space-between', py: 0.75 }}>
              <Skeleton variant="text" width={80} />
              <Skeleton variant="text" width={60} />
            </Box>
          ))}
        </Paper>
      </Grid>
    ))}
  </Grid>
);

function QuantStatsReport({ data, loading }) {
  if (loading) {
    return <LoadingSkeleton />;
  }

  if (!data) {
    return (
      <Typography variant="body2" sx={{ color: 'text.secondary', fontStyle: 'italic' }}>
        QuantStats not available
      </Typography>
    );
  }

  // Extract metrics from data - support both flat structure and nested structure
  const quantStats = data.quantstats || data.quant_stats || data;

  // Returns metrics
  const totalReturn = quantStats.total_return ?? quantStats['Total Return'] ?? null;
  const cagr = quantStats.cagr ?? quantStats['CAGR'] ?? null;
  const bestDay = quantStats.best_day ?? quantStats['Best Day'] ?? null;
  const worstDay = quantStats.worst_day ?? quantStats['Worst Day'] ?? null;
  const avgReturn = quantStats.avg_return ?? quantStats['Avg Return'] ?? null;
  const mtd = quantStats.mtd ?? quantStats['MTD'] ?? null;
  const ytd = quantStats.ytd ?? quantStats['YTD'] ?? null;

  // Risk metrics
  const volatility = quantStats.volatility ?? quantStats['Volatility (ann.)'] ?? quantStats.annual_volatility ?? null;
  const maxDrawdown = quantStats.max_drawdown ?? quantStats['Max Drawdown'] ?? null;
  const valueAtRisk = quantStats.var ?? quantStats['Value at Risk'] ?? quantStats.daily_var ?? null;
  const cvar = quantStats.cvar ?? quantStats['CVaR'] ?? quantStats.expected_shortfall ?? null;
  const avgDrawdown = quantStats.avg_drawdown ?? quantStats['Avg Drawdown'] ?? null;
  const recoveryFactor = quantStats.recovery_factor ?? quantStats['Recovery Factor'] ?? null;

  // Ratio metrics
  const sharpe = quantStats.sharpe ?? quantStats['Sharpe'] ?? quantStats.sharpe_ratio ?? null;
  const sortino = quantStats.sortino ?? quantStats['Sortino'] ?? quantStats.sortino_ratio ?? null;
  const calmar = quantStats.calmar ?? quantStats['Calmar'] ?? quantStats.calmar_ratio ?? null;
  const omega = quantStats.omega ?? quantStats['Omega'] ?? null;
  const profitFactor = quantStats.profit_factor ?? quantStats['Profit Factor'] ?? null;
  const payoffRatio = quantStats.payoff_ratio ?? quantStats['Payoff Ratio'] ?? null;

  return (
    <Grid container spacing={2}>
      {/* Returns Section */}
      <Grid item xs={12} md={4}>
        <Paper sx={paperSx}>
          <SectionHeader icon={TrendingUpIcon} title="Returns" />
          <Divider sx={{ mb: 1 }} />
          <MetricRow
            label="Total Return"
            value={formatPercent(totalReturn)}
            isPositive={totalReturn !== null ? totalReturn > 0 : undefined}
          />
          <MetricRow
            label="CAGR"
            value={formatPercent(cagr)}
            isPositive={cagr !== null ? cagr > 0 : undefined}
          />
          <MetricRow
            label="Best Day"
            value={formatPercent(bestDay)}
            isPositive={true}
          />
          <MetricRow
            label="Worst Day"
            value={formatPercent(worstDay)}
            isPositive={false}
          />
          {avgReturn !== null && (
            <MetricRow
              label="Avg Return"
              value={formatPercent(avgReturn)}
              isPositive={avgReturn > 0}
            />
          )}
          {mtd !== null && (
            <MetricRow
              label="MTD"
              value={formatPercent(mtd)}
              isPositive={mtd > 0}
            />
          )}
          {ytd !== null && (
            <MetricRow
              label="YTD"
              value={formatPercent(ytd)}
              isPositive={ytd > 0}
            />
          )}
        </Paper>
      </Grid>

      {/* Risk Section */}
      <Grid item xs={12} md={4}>
        <Paper sx={paperSx}>
          <SectionHeader icon={WarningIcon} title="Risk" />
          <Divider sx={{ mb: 1 }} />
          <MetricRow
            label="Volatility"
            value={formatPercent(volatility)}
          />
          <MetricRow
            label="Max Drawdown"
            value={formatPercent(maxDrawdown)}
            isPositive={false}
          />
          <MetricRow
            label="VaR (95%)"
            value={formatPercent(valueAtRisk)}
          />
          <MetricRow
            label="CVaR (95%)"
            value={formatPercent(cvar)}
          />
          {avgDrawdown !== null && (
            <MetricRow
              label="Avg Drawdown"
              value={formatPercent(avgDrawdown)}
              isPositive={false}
            />
          )}
          {recoveryFactor !== null && (
            <MetricRow
              label="Recovery Factor"
              value={formatDecimal(recoveryFactor)}
              isPositive={recoveryFactor > 1}
            />
          )}
        </Paper>
      </Grid>

      {/* Ratios Section */}
      <Grid item xs={12} md={4}>
        <Paper sx={paperSx}>
          <SectionHeader icon={BalanceIcon} title="Ratios" />
          <Divider sx={{ mb: 1 }} />
          <MetricRow
            label="Sharpe"
            value={formatDecimal(sharpe, 2)}
            isPositive={sharpe !== null ? sharpe > 0 : undefined}
          />
          <MetricRow
            label="Sortino"
            value={formatDecimal(sortino, 2)}
            isPositive={sortino !== null ? sortino > 0 : undefined}
          />
          <MetricRow
            label="Calmar"
            value={formatDecimal(calmar, 2)}
            isPositive={calmar !== null ? calmar > 0 : undefined}
          />
          <MetricRow
            label="Omega"
            value={formatDecimal(omega, 2)}
            isPositive={omega !== null ? omega > 1 : undefined}
          />
          {profitFactor !== null && (
            <MetricRow
              label="Profit Factor"
              value={formatDecimal(profitFactor, 2)}
              isPositive={profitFactor > 1}
            />
          )}
          {payoffRatio !== null && (
            <MetricRow
              label="Payoff Ratio"
              value={formatDecimal(payoffRatio, 2)}
              isPositive={payoffRatio > 1}
            />
          )}
        </Paper>
      </Grid>
    </Grid>
  );
}

QuantStatsReport.propTypes = {
  data: PropTypes.object,
  loading: PropTypes.bool,
};

QuantStatsReport.defaultProps = {
  data: null,
  loading: false,
};

export default QuantStatsReport;
