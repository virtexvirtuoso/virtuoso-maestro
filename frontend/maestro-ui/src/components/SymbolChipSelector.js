// SymbolChipSelector - Clickable chip grid for symbol selection
import React from 'react';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Skeleton from '@mui/material/Skeleton';
import Typography from '@mui/material/Typography';

/**
 * SymbolChipSelector - Renders all symbols for a provider as clickable chips
 *
 * @param {string} provider - Selected provider (used to show empty state message)
 * @param {string} value - Currently selected symbol
 * @param {function} onChange - Callback when symbol is selected
 * @param {boolean} disabled - Disable all chips
 * @param {string[]} symbols - Array of available symbols
 * @param {boolean} loading - Show loading skeleton state
 */
export default function SymbolChipSelector({
  provider,
  value,
  onChange,
  disabled = false,
  symbols = [],
  loading = false,
}) {
  const handleChipClick = (symbol) => {
    if (!disabled) {
      onChange(symbol);
    }
  };

  // Loading state - show skeleton chips
  if (loading) {
    return (
      <Box
        sx={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 1,
          p: 1,
        }}
      >
        {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
          <Skeleton
            key={i}
            variant="rounded"
            width={72}
            height={32}
            sx={{ borderRadius: '16px' }}
          />
        ))}
      </Box>
    );
  }

  // Empty state - no provider selected
  if (!provider) {
    return (
      <Box
        sx={{
          p: 2,
          textAlign: 'center',
          color: 'text.secondary',
          border: '1px dashed',
          borderColor: 'divider',
          borderRadius: 1,
        }}
      >
        <Typography variant="body2">
          Select a provider first
        </Typography>
      </Box>
    );
  }

  // Empty state - no symbols available
  if (symbols.length === 0) {
    return (
      <Box
        sx={{
          p: 2,
          textAlign: 'center',
          color: 'text.secondary',
          border: '1px dashed',
          borderColor: 'divider',
          borderRadius: 1,
        }}
      >
        <Typography variant="body2">
          No symbols available for {provider}
        </Typography>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: 1,
        p: 1,
        // Scroll if more than 12 symbols (roughly 3 rows)
        maxHeight: symbols.length > 12 ? 160 : 'auto',
        overflowY: symbols.length > 12 ? 'auto' : 'visible',
        // Custom scrollbar styling
        '&::-webkit-scrollbar': {
          width: 6,
        },
        '&::-webkit-scrollbar-track': {
          bgcolor: 'action.hover',
          borderRadius: 3,
        },
        '&::-webkit-scrollbar-thumb': {
          bgcolor: 'primary.main',
          borderRadius: 3,
        },
      }}
    >
      {symbols.map((symbol) => (
        <Chip
          key={symbol}
          label={symbol.toUpperCase()}
          variant={value === symbol ? 'filled' : 'outlined'}
          color={value === symbol ? 'primary' : 'default'}
          onClick={() => handleChipClick(symbol)}
          disabled={disabled}
          sx={{
            cursor: disabled ? 'not-allowed' : 'pointer',
            fontWeight: value === symbol ? 600 : 400,
            transition: 'all 0.2s ease',
            '&:hover': {
              transform: disabled ? 'none' : 'scale(1.05)',
            },
          }}
        />
      ))}
    </Box>
  );
}
