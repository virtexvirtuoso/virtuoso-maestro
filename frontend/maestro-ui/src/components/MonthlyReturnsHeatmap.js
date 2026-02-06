// Monthly returns heatmap with green/red gradients based on return values
import React, { useMemo } from 'react';
import PropTypes from 'prop-types';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Tooltip from '@mui/material/Tooltip';
import Skeleton from '@mui/material/Skeleton';

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

// Color gradients for positive (green) and negative (red) returns
const getColor = (value, maxAbs) => {
  if (value === null || value === undefined || isNaN(value)) {
    return 'rgba(255, 255, 255, 0.05)';
  }

  // Normalize value to 0-1 range based on maxAbs
  const intensity = Math.min(Math.abs(value) / maxAbs, 1);

  if (value >= 0) {
    // Green gradient: darker green for higher positive returns
    const r = Math.round(46 - intensity * 26);
    const g = Math.round(125 + intensity * 75);
    const b = Math.round(50 - intensity * 25);
    return `rgba(${r}, ${g}, ${b}, ${0.3 + intensity * 0.6})`;
  } else {
    // Red gradient: darker red for larger negative returns
    const r = Math.round(180 + intensity * 55);
    const g = Math.round(60 - intensity * 35);
    const b = Math.round(60 - intensity * 35);
    return `rgba(${r}, ${g}, ${b}, ${0.3 + intensity * 0.6})`;
  }
};

const getTextColor = (value, maxAbs) => {
  if (value === null || value === undefined || isNaN(value)) {
    return 'text.disabled';
  }
  const intensity = Math.min(Math.abs(value) / maxAbs, 1);
  return intensity > 0.4 ? 'rgba(255, 255, 255, 0.95)' : 'text.primary';
};

const formatPercent = (value) => {
  if (value === null || value === undefined || isNaN(value)) return '-';
  const percent = value * 100;
  return `${percent >= 0 ? '+' : ''}${percent.toFixed(1)}%`;
};

const formatExactPercent = (value) => {
  if (value === null || value === undefined || isNaN(value)) return 'No data';
  const percent = value * 100;
  return `${percent >= 0 ? '+' : ''}${percent.toFixed(4)}%`;
};

const LoadingSkeleton = () => (
  <Box>
    <Box sx={{ display: 'flex', gap: 0.5, mb: 0.5 }}>
      <Box sx={{ width: 48 }} />
      {MONTHS.map((m) => (
        <Skeleton key={m} variant="rectangular" width={48} height={24} />
      ))}
    </Box>
    {[0, 1, 2].map((i) => (
      <Box key={i} sx={{ display: 'flex', gap: 0.5, mb: 0.5 }}>
        <Skeleton variant="rectangular" width={48} height={36} />
        {MONTHS.map((_, j) => (
          <Skeleton key={j} variant="rectangular" width={48} height={36} />
        ))}
      </Box>
    ))}
  </Box>
);

function MonthlyReturnsHeatmap({ data, loading }) {
  // Process data into year -> month -> value structure
  const { heatmapData, years, maxAbs } = useMemo(() => {
    if (!data || (Array.isArray(data) && data.length === 0) ||
        (typeof data === 'object' && Object.keys(data).length === 0)) {
      return { heatmapData: {}, years: [], maxAbs: 0.1 };
    }

    const processed = {};
    let maxAbsValue = 0;

    // Handle array format: [{year, month, return}, ...]
    if (Array.isArray(data)) {
      data.forEach((item) => {
        const year = item.year;
        const month = item.month; // 1-indexed
        const returnValue = item.return ?? item.value ?? item.returns;

        if (!processed[year]) {
          processed[year] = {};
        }
        processed[year][month] = returnValue;

        if (returnValue !== null && returnValue !== undefined && !isNaN(returnValue)) {
          maxAbsValue = Math.max(maxAbsValue, Math.abs(returnValue));
        }
      });
    }
    // Handle nested object format: {2023: {1: 0.05, 2: -0.02}, ...}
    else if (typeof data === 'object') {
      Object.entries(data).forEach(([year, months]) => {
        processed[year] = {};
        if (typeof months === 'object') {
          Object.entries(months).forEach(([month, value]) => {
            processed[year][parseInt(month)] = value;
            if (value !== null && value !== undefined && !isNaN(value)) {
              maxAbsValue = Math.max(maxAbsValue, Math.abs(value));
            }
          });
        }
      });
    }

    const sortedYears = Object.keys(processed).sort((a, b) => parseInt(b) - parseInt(a));

    return {
      heatmapData: processed,
      years: sortedYears,
      maxAbs: maxAbsValue || 0.1, // Prevent division by zero
    };
  }, [data]);

  if (loading) {
    return <LoadingSkeleton />;
  }

  if (years.length === 0) {
    return (
      <Typography variant="body2" sx={{ color: 'text.secondary', fontStyle: 'italic' }}>
        Monthly returns data not available
      </Typography>
    );
  }

  return (
    <Box sx={{ overflowX: 'auto' }}>
      {/* Header row with month labels */}
      <Box sx={{ display: 'flex', gap: 0.5, mb: 0.5, minWidth: 'fit-content' }}>
        <Box
          sx={{
            width: 56,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Typography
            variant="caption"
            sx={{ fontFamily: '"IBM Plex Mono", monospace', fontWeight: 600, color: 'text.secondary' }}
          >
            Year
          </Typography>
        </Box>
        {MONTHS.map((month) => (
          <Box
            key={month}
            sx={{
              width: 52,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              py: 0.5,
            }}
          >
            <Typography
              variant="caption"
              sx={{ fontFamily: '"IBM Plex Mono", monospace', fontWeight: 600, color: 'text.secondary' }}
            >
              {month}
            </Typography>
          </Box>
        ))}
        <Box
          sx={{
            width: 60,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Typography
            variant="caption"
            sx={{ fontFamily: '"IBM Plex Mono", monospace', fontWeight: 600, color: 'text.secondary' }}
          >
            Total
          </Typography>
        </Box>
      </Box>

      {/* Data rows */}
      {years.map((year) => {
        const yearData = heatmapData[year] || {};

        // Calculate yearly total
        const yearlyTotal = Object.values(yearData).reduce((sum, val) => {
          if (val !== null && val !== undefined && !isNaN(val)) {
            return sum + val;
          }
          return sum;
        }, 0);

        return (
          <Box key={year} sx={{ display: 'flex', gap: 0.5, mb: 0.5, minWidth: 'fit-content' }}>
            {/* Year label */}
            <Box
              sx={{
                width: 56,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                borderRadius: 0.5,
              }}
            >
              <Typography
                variant="caption"
                sx={{ fontFamily: '"IBM Plex Mono", monospace', fontWeight: 600 }}
              >
                {year}
              </Typography>
            </Box>

            {/* Monthly cells */}
            {MONTHS.map((_, monthIndex) => {
              const month = monthIndex + 1;
              const value = yearData[month];
              const bgColor = getColor(value, maxAbs);
              const textColor = getTextColor(value, maxAbs);

              return (
                <Tooltip
                  key={month}
                  title={
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="caption" sx={{ fontWeight: 600 }}>
                        {MONTHS[monthIndex]} {year}
                      </Typography>
                      <br />
                      <Typography variant="caption">
                        {formatExactPercent(value)}
                      </Typography>
                    </Box>
                  }
                  arrow
                  placement="top"
                >
                  <Box
                    sx={{
                      width: 52,
                      height: 36,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      backgroundColor: bgColor,
                      borderRadius: 0.5,
                      cursor: 'default',
                      transition: 'transform 0.15s ease-in-out, box-shadow 0.15s ease-in-out',
                      '&:hover': {
                        transform: 'scale(1.05)',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
                        zIndex: 1,
                      },
                    }}
                  >
                    <Typography
                      variant="caption"
                      sx={{
                        fontFamily: '"IBM Plex Mono", monospace',
                        fontWeight: 500,
                        fontSize: '0.7rem',
                        color: textColor,
                      }}
                    >
                      {formatPercent(value)}
                    </Typography>
                  </Box>
                </Tooltip>
              );
            })}

            {/* Yearly total */}
            <Tooltip
              title={
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="caption" sx={{ fontWeight: 600 }}>
                    {year} Total
                  </Typography>
                  <br />
                  <Typography variant="caption">
                    {formatExactPercent(yearlyTotal)}
                  </Typography>
                </Box>
              }
              arrow
              placement="top"
            >
              <Box
                sx={{
                  width: 60,
                  height: 36,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  backgroundColor: getColor(yearlyTotal, maxAbs * 3), // Use wider scale for totals
                  borderRadius: 0.5,
                  border: '1px solid',
                  borderColor: 'divider',
                  cursor: 'default',
                }}
              >
                <Typography
                  variant="caption"
                  sx={{
                    fontFamily: '"IBM Plex Mono", monospace',
                    fontWeight: 600,
                    fontSize: '0.7rem',
                    color: getTextColor(yearlyTotal, maxAbs * 3),
                  }}
                >
                  {formatPercent(yearlyTotal)}
                </Typography>
              </Box>
            </Tooltip>
          </Box>
        );
      })}

      {/* Legend */}
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', mt: 2, gap: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <Box
            sx={{
              width: 12,
              height: 12,
              backgroundColor: 'rgba(235, 25, 25, 0.9)',
              borderRadius: 0.25,
            }}
          />
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            Negative
          </Typography>
        </Box>
        <Box
          sx={{
            width: 80,
            height: 8,
            background: 'linear-gradient(to right, rgba(235, 25, 25, 0.9), rgba(255, 255, 255, 0.1), rgba(20, 200, 25, 0.9))',
            borderRadius: 0.5,
          }}
        />
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <Box
            sx={{
              width: 12,
              height: 12,
              backgroundColor: 'rgba(20, 200, 25, 0.9)',
              borderRadius: 0.25,
            }}
          />
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            Positive
          </Typography>
        </Box>
      </Box>
    </Box>
  );
}

MonthlyReturnsHeatmap.propTypes = {
  data: PropTypes.oneOfType([PropTypes.array, PropTypes.object]),
  loading: PropTypes.bool,
};

MonthlyReturnsHeatmap.defaultProps = {
  data: null,
  loading: false,
};

export default MonthlyReturnsHeatmap;
