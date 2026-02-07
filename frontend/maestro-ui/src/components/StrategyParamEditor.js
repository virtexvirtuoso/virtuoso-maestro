// StrategyParamEditor - Enhanced parameter inputs with descriptions, ranges, and validation
// Fetches param metadata from /strategy/{name}/params/metadata endpoint
import React, { useState, useEffect, useCallback } from 'react';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import TextField from '@mui/material/TextField';
import Slider from '@mui/material/Slider';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import Skeleton from '@mui/material/Skeleton';
import Alert from '@mui/material/Alert';
import InputAdornment from '@mui/material/InputAdornment';
import IconButton from '@mui/material/IconButton';
import Collapse from '@mui/material/Collapse';
import TuneIcon from '@mui/icons-material/Tune';

/**
 * StrategyParamEditor - Enhanced parameter editor with metadata
 *
 * @param {string} strategy - Strategy name to fetch params for
 * @param {object} values - Current parameter values {paramName: value}
 * @param {function} onChange - Callback when any param changes (paramName, value)
 * @param {boolean} showSliders - Whether to show sliders for bounded params (default: true)
 */
export default function StrategyParamEditor({
  strategy,
  values,
  onChange,
  showSliders = true,
}) {
  const [metadata, setMetadata] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expandedSliders, setExpandedSliders] = useState({});

  // Fetch parameter metadata when strategy changes
  const fetchMetadata = useCallback(async () => {
    if (!strategy) {
      setMetadata({});
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${process.env.REACT_APP_REST_API_URL}/strategy/${strategy}/params/metadata`
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch param metadata: ${response.status}`);
      }

      const data = await response.json();
      setMetadata(data);

      // Initialize slider expansion state
      const sliderState = {};
      Object.keys(data).forEach((param) => {
        sliderState[param] = false;
      });
      setExpandedSliders(sliderState);
    } catch (err) {
      console.error('Failed to fetch strategy param metadata:', err);
      setError(err.message);
      // Fallback: still allow editing without metadata
      setMetadata({});
    } finally {
      setLoading(false);
    }
  }, [strategy]);

  useEffect(() => {
    fetchMetadata();
  }, [fetchMetadata]);

  // Toggle slider visibility for a param
  const toggleSlider = (param) => {
    setExpandedSliders((prev) => ({
      ...prev,
      [param]: !prev[param],
    }));
  };

  // Validate value against min/max
  const validateValue = (param, value, meta) => {
    if (meta.min !== null && value < meta.min) {
      return { valid: false, error: `Min: ${meta.min}` };
    }
    if (meta.max !== null && value > meta.max) {
      return { valid: false, error: `Max: ${meta.max}` };
    }
    return { valid: true, error: null };
  };

  // Handle text field change
  const handleInputChange = (param, rawValue) => {
    const meta = metadata[param] || {};
    const value = meta.type === 'float' ? parseFloat(rawValue) : parseInt(rawValue, 10);

    if (!isNaN(value)) {
      onChange(param, value);
    }
  };

  // Handle slider change
  const handleSliderChange = (param, value) => {
    onChange(param, value);
  };

  // Get parameter list (from metadata or values fallback)
  const params = Object.keys(metadata).length > 0 ? Object.keys(metadata) : Object.keys(values);

  if (loading) {
    return (
      <Box>
        {[1, 2, 3].map((i) => (
          <Skeleton key={i} height={72} sx={{ mb: 1 }} />
        ))}
      </Box>
    );
  }

  if (params.length === 0) {
    return (
      <Alert severity="info">This strategy has no configurable parameters</Alert>
    );
  }

  return (
    <Box>
      {error && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          Could not load parameter descriptions. Using defaults.
        </Alert>
      )}

      <Grid container spacing={2}>
        {params.map((param) => {
          const meta = metadata[param] || {};
          const currentValue = values[param] ?? meta.value ?? 0;
          const validation = validateValue(param, currentValue, meta);
          const hasBounds = meta.min !== null && meta.max !== null;
          const showSlider = showSliders && hasBounds && expandedSliders[param];

          // Build tooltip content
          const tooltipContent = (
            <Box>
              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                {param}
              </Typography>
              {meta.description && (
                <Typography variant="body2">{meta.description}</Typography>
              )}
              {hasBounds && (
                <Typography variant="caption" color="text.secondary">
                  Range: {meta.min} - {meta.max}
                  {meta.step && ` (step: ${meta.step})`}
                </Typography>
              )}
              {meta.value !== undefined && (
                <Typography variant="caption" display="block" color="text.secondary">
                  Default: {meta.value}
                </Typography>
              )}
            </Box>
          );

          return (
            <Grid item xs={12} sm={6} md={4} key={param}>
              <Box>
                <Tooltip title={tooltipContent} arrow placement="top">
                  <TextField
                    fullWidth
                    label={formatParamLabel(param)}
                    type="number"
                    value={currentValue}
                    onChange={(e) => handleInputChange(param, e.target.value)}
                    error={!validation.valid}
                    helperText={
                      validation.error ||
                      meta.description ||
                      `Default: ${meta.value ?? 'N/A'}`
                    }
                    inputProps={{
                      min: meta.min ?? undefined,
                      max: meta.max ?? undefined,
                      step: meta.step ?? (meta.type === 'float' ? 0.1 : 1),
                    }}
                    InputProps={{
                      endAdornment: hasBounds && showSliders && (
                        <InputAdornment position="end">
                          <IconButton
                            size="small"
                            onClick={() => toggleSlider(param)}
                            color={showSlider ? 'primary' : 'default'}
                            title="Toggle slider"
                          >
                            <TuneIcon fontSize="small" />
                          </IconButton>
                        </InputAdornment>
                      ),
                    }}
                    sx={{
                      '& .MuiFormHelperText-root': {
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      },
                    }}
                  />
                </Tooltip>

                {/* Slider for bounded params */}
                <Collapse in={showSlider}>
                  <Box sx={{ px: 1, pt: 1 }}>
                    <Slider
                      value={currentValue}
                      onChange={(_, value) => handleSliderChange(param, value)}
                      min={meta.min}
                      max={meta.max}
                      step={meta.step ?? (meta.type === 'float' ? 0.1 : 1)}
                      marks={[
                        { value: meta.min, label: String(meta.min) },
                        { value: meta.max, label: String(meta.max) },
                      ]}
                      valueLabelDisplay="auto"
                      size="small"
                    />
                  </Box>
                </Collapse>
              </Box>
            </Grid>
          );
        })}
      </Grid>
    </Box>
  );
}

/**
 * Format parameter name for display
 * e.g., "senkou_b" -> "Senkou B", "fast" -> "Fast"
 */
function formatParamLabel(param) {
  return param
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}
