// TimeframeSelector - ToggleButtonGroup for quick timeframe selection
// Replaces dropdown with visual button group for better UX
import React from 'react';
import Box from '@mui/material/Box';
import ToggleButton from '@mui/material/ToggleButton';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import Tooltip from '@mui/material/Tooltip';
import Skeleton from '@mui/material/Skeleton';

// Default timeframe options (standard across exchanges)
const DEFAULT_TIMEFRAMES = [
  { value: '1m', label: '1m' },
  { value: '5m', label: '5m' },
  { value: '15m', label: '15m' },
  { value: '1h', label: '1h' },
  { value: '4h', label: '4h' },
  { value: '1d', label: '1d' },
];

/**
 * TimeframeSelector - Visual button group for selecting chart timeframes
 *
 * @param {string} value - Currently selected timeframe
 * @param {function} onChange - Callback when timeframe changes (receives value)
 * @param {string[]} availableTimeframes - Array of available timeframe values for current symbol
 * @param {boolean} loading - Show skeleton loading state
 * @param {boolean} disabled - Disable all buttons
 * @param {object[]} timeframes - Override default timeframe options [{value, label}]
 */
export default function TimeframeSelector({
  value,
  onChange,
  availableTimeframes = null, // null means all available
  loading = false,
  disabled = false,
  timeframes = DEFAULT_TIMEFRAMES,
}) {
  const handleChange = (event, newValue) => {
    // ToggleButtonGroup with exclusive sends null when clicking selected button
    // Keep current value in that case
    if (newValue !== null) {
      onChange(newValue);
    }
  };

  // Determine if a timeframe is available
  const isAvailable = (tf) => {
    if (availableTimeframes === null) return true; // All available
    return availableTimeframes.includes(tf);
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', gap: 0.5 }}>
        {timeframes.map((tf) => (
          <Skeleton
            key={tf.value}
            variant="rectangular"
            width={48}
            height={40}
            sx={{ borderRadius: 1 }}
          />
        ))}
      </Box>
    );
  }

  return (
    <ToggleButtonGroup
      value={value}
      exclusive
      onChange={handleChange}
      aria-label="timeframe selection"
      disabled={disabled}
      sx={{
        '& .MuiToggleButton-root': {
          px: 2,
          py: 1,
          minWidth: 48,
          textTransform: 'none',
          fontWeight: 500,
          '&.Mui-selected': {
            bgcolor: 'primary.main',
            color: 'primary.contrastText',
            fontWeight: 600,
            '&:hover': {
              bgcolor: 'primary.dark',
            },
          },
          '&.Mui-disabled': {
            color: 'text.disabled',
            bgcolor: 'action.disabledBackground',
          },
        },
      }}
    >
      {timeframes.map((tf) => {
        const available = isAvailable(tf.value);
        const button = (
          <ToggleButton
            key={tf.value}
            value={tf.value}
            disabled={!available || disabled}
            aria-label={tf.label}
          >
            {tf.label}
          </ToggleButton>
        );

        // Wrap disabled buttons with tooltip
        if (!available) {
          return (
            <Tooltip
              key={tf.value}
              title="Not available for this symbol"
              arrow
              placement="top"
            >
              <span>{button}</span>
            </Tooltip>
          );
        }

        return button;
      })}
    </ToggleButtonGroup>
  );
}
