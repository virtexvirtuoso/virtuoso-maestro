// WalkForwardSettings - Collapsible panel for Walk-Forward optimization configuration
// Shows when optType is WALKFORWARD or BOTH
import React from 'react';
import Box from '@mui/material/Box';
import Collapse from '@mui/material/Collapse';
import Grid from '@mui/material/Grid';
import TextField from '@mui/material/TextField';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import FormHelperText from '@mui/material/FormHelperText';
import Typography from '@mui/material/Typography';
import TuneIcon from '@mui/icons-material/Tune';

// Walk-forward mode options with descriptions
const WFO_MODES = [
  {
    value: 'rolling',
    label: 'Rolling (Recommended)',
    description: 'Fixed window size slides forward',
  },
  {
    value: 'expanding',
    label: 'Expanding',
    description: 'Training window grows with each split',
  },
  {
    value: 'adaptive',
    label: 'Adaptive',
    description: 'Window adjusts based on volatility',
  },
];

/**
 * WalkForwardSettings - Configures walk-forward optimization parameters
 *
 * @param {string} optType - Current optimization type (BACKTESTING, WALKFORWARD, BOTH)
 * @param {number} numSplits - Number of time-series splits (default: 10)
 * @param {string} mode - Walk-forward mode (rolling, expanding, adaptive)
 * @param {function} onNumSplitsChange - Callback when splits value changes
 * @param {function} onModeChange - Callback when mode changes
 */
export default function WalkForwardSettings({
  optType,
  numSplits = 10,
  mode = 'rolling',
  onNumSplitsChange,
  onModeChange,
}) {
  // Only visible when not pure backtesting
  const isVisible = optType === 'WALKFORWARD' || optType === 'BOTH';

  // Get current mode description for helper text
  const currentModeConfig = WFO_MODES.find((m) => m.value === mode) || WFO_MODES[0];

  return (
    <Collapse in={isVisible} timeout={300}>
      <Box sx={{ mt: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
          <TuneIcon color="primary" fontSize="small" />
          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
            Walk-Forward Settings
          </Typography>
        </Box>

        <Grid container spacing={2}>
          {/* Number of Splits */}
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Number of Splits"
              type="number"
              value={numSplits}
              onChange={(e) => onNumSplitsChange(Number(e.target.value))}
              helperText="More splits = more robust but slower"
              inputProps={{ min: 2, max: 50 }}
            />
          </Grid>

          {/* Walk-Forward Mode */}
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel id="wfo-mode-label">Mode</InputLabel>
              <Select
                labelId="wfo-mode-label"
                id="wfo-mode-select"
                value={mode}
                label="Mode"
                onChange={(e) => onModeChange(e.target.value)}
              >
                {WFO_MODES.map((m) => (
                  <MenuItem key={m.value} value={m.value}>
                    {m.label}
                  </MenuItem>
                ))}
              </Select>
              <FormHelperText>{currentModeConfig.description}</FormHelperText>
            </FormControl>
          </Grid>
        </Grid>
      </Box>
    </Collapse>
  );
}
