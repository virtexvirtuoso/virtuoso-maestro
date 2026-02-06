// Parameter Stability Chart - Shows parameter values across walk-forward folds
import React, { useMemo } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { useTheme } from '@mui/material/styles';

// Parameter line colors matching theme
const PARAM_COLORS = [
  '#fbbf24', // amber (primary)
  '#06B6D4', // cyan (secondary)
  '#10b981', // emerald/success
  '#ef4444', // red/error
  '#a855f7', // purple
  '#f97316', // orange
  '#14b8a6', // teal
  '#ec4899', // pink
];

/**
 * ParameterStabilityChart - Displays parameter values across walk-forward folds
 * using a Recharts LineChart. Shows how parameters drift or remain stable.
 *
 * @param {Object} props
 * @param {Array} props.walkforwardData - Array of WFO split results with parameters
 * @param {Object} props.parameterRanges - Optional object with parameter ranges for context
 */
export default function ParameterStabilityChart({ walkforwardData, parameterRanges }) {
  const theme = useTheme();

  // Transform data for Recharts
  const { chartData, parameterNames } = useMemo(() => {
    if (!walkforwardData || walkforwardData.length === 0) {
      return { chartData: [], parameterNames: [] };
    }

    // Sort by split number
    const sortedData = [...walkforwardData].sort((a, b) =>
      a.num_split < b.num_split ? -1 : a.num_split > b.num_split ? 1 : 0
    );

    // Extract parameter names from first split
    const firstSplit = sortedData[0];
    const params = firstSplit?.parameters ? Object.keys(firstSplit.parameters) : [];

    // Build chart data array
    const data = sortedData.map((split, index) => {
      const row = {
        fold: `Fold ${split.num_split ?? index}`,
        foldNum: split.num_split ?? index,
      };

      params.forEach((param) => {
        row[param] = split.parameters?.[param] ?? null;
      });

      return row;
    });

    return { chartData: data, parameterNames: params };
  }, [walkforwardData]);

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload || payload.length === 0) return null;

    return (
      <Box
        sx={{
          backgroundColor: theme.palette.background.paper,
          border: `1px solid ${theme.palette.divider}`,
          borderRadius: 1,
          p: 1.5,
          boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
        }}
      >
        <Typography
          variant="caption"
          sx={{
            fontWeight: 600,
            color: theme.palette.primary.main,
            fontFamily: '"IBM Plex Mono", monospace',
          }}
        >
          {label}
        </Typography>
        {payload.map((entry, index) => (
          <Box key={index} sx={{ display: 'flex', gap: 1, mt: 0.5 }}>
            <Box
              sx={{
                width: 10,
                height: 10,
                borderRadius: '50%',
                backgroundColor: entry.color,
                mt: 0.5,
              }}
            />
            <Typography variant="caption" color="text.secondary">
              {entry.name}:{' '}
              <span style={{ color: theme.palette.text.primary, fontWeight: 500 }}>
                {typeof entry.value === 'number' ? entry.value.toFixed(4) : entry.value}
              </span>
            </Typography>
          </Box>
        ))}
      </Box>
    );
  };

  if (chartData.length === 0 || parameterNames.length === 0) {
    return (
      <Typography variant="body2" color="text.secondary">
        No parameter data available for stability analysis
      </Typography>
    );
  }

  return (
    <Box sx={{ width: '100%', py: 2 }}>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke={theme.palette.divider}
            opacity={0.5}
          />
          <XAxis
            dataKey="fold"
            stroke={theme.palette.text.secondary}
            tick={{ fill: theme.palette.text.secondary, fontSize: 11 }}
            axisLine={{ stroke: theme.palette.divider }}
          />
          <YAxis
            stroke={theme.palette.text.secondary}
            tick={{ fill: theme.palette.text.secondary, fontSize: 11 }}
            axisLine={{ stroke: theme.palette.divider }}
            tickFormatter={(value) => (typeof value === 'number' ? value.toFixed(2) : value)}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{
              paddingTop: 20,
              fontFamily: '"IBM Plex Mono", monospace',
            }}
            iconType="circle"
            iconSize={8}
          />

          {parameterNames.map((param, index) => (
            <Line
              key={param}
              type="monotone"
              dataKey={param}
              name={param}
              stroke={PARAM_COLORS[index % PARAM_COLORS.length]}
              strokeWidth={2}
              dot={{
                fill: PARAM_COLORS[index % PARAM_COLORS.length],
                strokeWidth: 0,
                r: 4,
              }}
              activeDot={{
                r: 6,
                stroke: theme.palette.background.paper,
                strokeWidth: 2,
              }}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>

      {/* Parameter stability metrics */}
      {parameterNames.length > 0 && (
        <Box
          sx={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: 2,
            mt: 2,
            pt: 2,
            borderTop: `1px solid ${theme.palette.divider}`,
          }}
        >
          {parameterNames.map((param, index) => {
            // Calculate coefficient of variation for stability indicator
            const values = chartData.map((d) => d[param]).filter((v) => v !== null);
            const mean = values.reduce((a, b) => a + b, 0) / values.length;
            const stdDev = Math.sqrt(
              values.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / values.length
            );
            const cv = mean !== 0 ? (stdDev / Math.abs(mean)) * 100 : 0;

            // Stability classification
            let stabilityLabel = 'Stable';
            let stabilityColor = theme.palette.success.main;
            if (cv > 50) {
              stabilityLabel = 'Highly Variable';
              stabilityColor = theme.palette.error.main;
            } else if (cv > 20) {
              stabilityLabel = 'Moderate';
              stabilityColor = theme.palette.warning.main;
            }

            return (
              <Box
                key={param}
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1,
                  px: 1.5,
                  py: 0.75,
                  backgroundColor: 'rgba(255,255,255,0.03)',
                  borderRadius: 1,
                  border: `1px solid ${theme.palette.divider}`,
                }}
              >
                <Box
                  sx={{
                    width: 10,
                    height: 10,
                    borderRadius: '50%',
                    backgroundColor: PARAM_COLORS[index % PARAM_COLORS.length],
                  }}
                />
                <Typography
                  variant="caption"
                  sx={{ fontFamily: '"IBM Plex Mono", monospace' }}
                >
                  {param}:
                </Typography>
                <Typography
                  variant="caption"
                  sx={{
                    color: stabilityColor,
                    fontWeight: 600,
                    fontFamily: '"IBM Plex Mono", monospace',
                  }}
                >
                  {stabilityLabel}
                </Typography>
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ fontFamily: '"IBM Plex Mono", monospace' }}
                >
                  (CV: {cv.toFixed(1)}%)
                </Typography>
              </Box>
            );
          })}
        </Box>
      )}
    </Box>
  );
}
