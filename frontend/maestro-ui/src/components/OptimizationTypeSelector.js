// OptimizationTypeSelector - Visual cards for selecting optimization type
// Replaces radio buttons with descriptive cards showing type, description, and icon
import React from 'react';
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardActionArea from '@mui/material/CardActionArea';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Chip from '@mui/material/Chip';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import TimelineIcon from '@mui/icons-material/Timeline';
import AllInclusiveIcon from '@mui/icons-material/AllInclusive';

// Optimization type configuration with descriptions and icons
const OPTIMIZATION_TYPES = [
  {
    value: 'BACKTESTING',
    label: 'Backtest',
    description: 'Single pass with fixed parameters',
    icon: PlayArrowIcon,
    recommended: false,
  },
  {
    value: 'WALKFORWARD',
    label: 'Walk-Forward',
    description: 'Rolling optimization (recommended)',
    icon: TimelineIcon,
    recommended: true,
  },
  {
    value: 'BOTH',
    label: 'Both',
    description: 'Full analysis - backtest + walk-forward',
    icon: AllInclusiveIcon,
    recommended: false,
  },
];

// Gold theme for selected state (uses Maestro palette)
const SELECTED_COLORS = {
  border: '#fbbf24',      // primary.main (amber gold)
  background: 'rgba(251, 191, 36, 0.08)',  // Light gold tint
  iconColor: '#fbbf24',   // primary.main
};

export default function OptimizationTypeSelector({
  value,
  onChange,
  disabled = false,
}) {
  const handleSelect = (optValue) => {
    if (!disabled && onChange) {
      onChange(optValue);
    }
  };

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: { xs: 'column', sm: 'row' },
        gap: 2,
      }}
    >
      {OPTIMIZATION_TYPES.map((opt) => {
        const isSelected = value === opt.value;
        const IconComponent = opt.icon;

        return (
          <Card
            key={opt.value}
            sx={{
              flex: 1,
              minWidth: { xs: 'auto', sm: 180 },
              border: isSelected ? `2px solid ${SELECTED_COLORS.border}` : '1px solid',
              borderColor: isSelected ? SELECTED_COLORS.border : 'divider',
              bgcolor: isSelected ? SELECTED_COLORS.background : 'background.paper',
              opacity: disabled ? 0.6 : 1,
              transition: 'all 0.2s ease-in-out',
              '&:hover': disabled ? {} : {
                boxShadow: 3,
                borderColor: isSelected ? SELECTED_COLORS.border : 'primary.main',
              },
            }}
          >
            <CardActionArea
              onClick={() => handleSelect(opt.value)}
              disabled={disabled}
              sx={{ height: '100%' }}
            >
              <CardContent sx={{ p: 2 }}>
                {/* Icon */}
                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'center',
                    mb: 1.5,
                  }}
                >
                  <IconComponent
                    sx={{
                      fontSize: 40,
                      color: isSelected ? SELECTED_COLORS.iconColor : 'text.secondary',
                      transition: 'color 0.2s',
                    }}
                  />
                </Box>

                {/* Label with optional recommended badge */}
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: 1,
                    mb: 1,
                  }}
                >
                  <Typography
                    variant="subtitle1"
                    sx={{
                      fontWeight: isSelected ? 700 : 600,
                      textAlign: 'center',
                      color: isSelected ? 'text.primary' : 'text.primary',
                    }}
                  >
                    {opt.label}
                  </Typography>
                  {opt.recommended && (
                    <Chip
                      label="Recommended"
                      size="small"
                      sx={{
                        height: 20,
                        fontSize: '0.65rem',
                        fontWeight: 600,
                        bgcolor: isSelected ? SELECTED_COLORS.border : 'primary.main',
                        color: 'white',
                      }}
                    />
                  )}
                </Box>

                {/* Description */}
                <Typography
                  variant="body2"
                  color="text.secondary"
                  sx={{
                    textAlign: 'center',
                    lineHeight: 1.4,
                  }}
                >
                  {opt.description}
                </Typography>
              </CardContent>
            </CardActionArea>
          </Card>
        );
      })}
    </Box>
  );
}
