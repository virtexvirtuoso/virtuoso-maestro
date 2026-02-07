// StrategyCategorySelector - Tabbed category view for strategy selection
// Displays 6 category tabs with strategy list below, search, and selection highlighting
import React, { useState, useMemo, useCallback } from 'react';
import Box from '@mui/material/Box';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import TextField from '@mui/material/TextField';
import InputAdornment from '@mui/material/InputAdornment';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemText from '@mui/material/ListItemText';
import Typography from '@mui/material/Typography';
import Skeleton from '@mui/material/Skeleton';
import Paper from '@mui/material/Paper';
import Chip from '@mui/material/Chip';
import SearchIcon from '@mui/icons-material/Search';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

// Strategy categories with counts (65 total)
const STRATEGY_CATEGORIES = [
  { key: 'technical', name: 'Technical', count: 19 },
  { key: 'hybrids', name: 'Hybrids', count: 15 },
  { key: 'derivatives', name: 'Derivatives', count: 11 },
  { key: 'scalping', name: 'Scalping', count: 8 },
  { key: 'momentum', name: 'Momentum', count: 6 },
  { key: 'composite', name: 'Composite', count: 6 },
];

// Default descriptions for strategies without API data
const DEFAULT_DESCRIPTIONS = {
  // Technical
  'EMA_Cross': 'Exponential moving average crossover signals',
  'MACD': 'Moving average convergence divergence indicator',
  'RSI': 'Relative strength index momentum oscillator',
  'Bollinger_Bands': 'Volatility bands around moving average',
  'Ichimoku': 'Multi-component trend and momentum system',
  'ADX': 'Average directional index trend strength',
  'Stochastic': 'Stochastic oscillator momentum signals',
  'CCI': 'Commodity channel index deviation signals',
  'Williams_R': 'Williams %R overbought/oversold indicator',
  'ATR_Breakout': 'Average true range volatility breakout',
  // Scalping
  'VWAP': 'Volume weighted average price mean reversion',
  'StochRSI': 'Stochastic RSI fast momentum signals',
  'EMA_Ribbon': 'Multiple EMA ribbon trend following',
  // Momentum
  'TSMOM': 'Time-series momentum strategy',
  'Momentum': 'Price momentum trend following',
  // Derivatives
  'OI_Delta': 'Open interest change analysis',
  'CVD': 'Cumulative volume delta divergence',
  'Funding_Rate': 'Perpetual funding rate arbitrage',
  'Liquidation': 'Liquidation cascade detection',
  // Composite
  'Fernando': 'Multi-indicator confluence strategy',
  'Combined': 'Combined signal aggregation',
};

export default function StrategyCategorySelector({
  value,
  onChange,
  loading = false,
  strategies = [],
  strategyDescriptions = {},
}) {
  const [activeCategory, setActiveCategory] = useState('technical');
  const [searchQuery, setSearchQuery] = useState('');
  const [loadingDescriptions, setLoadingDescriptions] = useState({});
  const [fetchedDescriptions, setFetchedDescriptions] = useState({});

  // Parse strategies into category structure
  const categorizedStrategies = useMemo(() => {
    const result = {};
    STRATEGY_CATEGORIES.forEach((cat) => {
      result[cat.key] = [];
    });

    // If we have real strategy data with categories
    if (strategies.length > 0) {
      strategies.forEach((strat) => {
        if (typeof strat === 'object' && strat.category) {
          const cat = strat.category.toLowerCase();
          if (result[cat]) {
            result[cat].push(strat);
          }
        } else if (typeof strat === 'string') {
          // Heuristic categorization for string-only data
          const name = strat.toLowerCase();
          if (name.includes('oi') || name.includes('cvd') || name.includes('funding') ||
              name.includes('liquidation') || name.includes('basis') || name.includes('perp')) {
            result.derivatives.push({ name: strat, category: 'derivatives' });
          } else if (name.includes('vwap') || name.includes('stochrsi') || name.includes('ribbon') ||
                     name.includes('scalp') || name.includes('tick')) {
            result.scalping.push({ name: strat, category: 'scalping' });
          } else if (name.includes('momentum') || name.includes('tsmom') || name.includes('velocity')) {
            result.momentum.push({ name: strat, category: 'momentum' });
          } else if (name.includes('fernando') || name.includes('combined') || name.includes('multi') ||
                     name.includes('composite')) {
            result.composite.push({ name: strat, category: 'composite' });
          } else if (name.includes('+') || name.includes('hybrid') || name.includes('filter')) {
            result.hybrids.push({ name: strat, category: 'hybrids' });
          } else {
            result.technical.push({ name: strat, category: 'technical' });
          }
        }
      });
    }

    return result;
  }, [strategies]);

  // Fetch description for a strategy
  const fetchDescription = useCallback(async (strategyName) => {
    if (fetchedDescriptions[strategyName] || loadingDescriptions[strategyName]) {
      return;
    }

    setLoadingDescriptions((prev) => ({ ...prev, [strategyName]: true }));

    try {
      const response = await fetch(
        `${process.env.REACT_APP_REST_API_URL}/strategy/${strategyName}/description`
      );
      if (response.ok) {
        const data = await response.json();
        setFetchedDescriptions((prev) => ({
          ...prev,
          [strategyName]: data.description || data,
        }));
      }
    } catch {
      // Use default description on error
    } finally {
      setLoadingDescriptions((prev) => ({ ...prev, [strategyName]: false }));
    }
  }, [fetchedDescriptions, loadingDescriptions]);

  // Get description for a strategy
  const getDescription = useCallback((strategyName) => {
    // Check prop-provided descriptions first
    if (strategyDescriptions[strategyName]) {
      return strategyDescriptions[strategyName];
    }
    // Check fetched descriptions
    if (fetchedDescriptions[strategyName]) {
      return fetchedDescriptions[strategyName];
    }
    // Check default descriptions
    if (DEFAULT_DESCRIPTIONS[strategyName]) {
      return DEFAULT_DESCRIPTIONS[strategyName];
    }
    // Generic description
    return 'Trading strategy';
  }, [strategyDescriptions, fetchedDescriptions]);

  // Filter strategies by search query
  const filteredStrategies = useMemo(() => {
    const categoryStrategies = categorizedStrategies[activeCategory] || [];

    if (!searchQuery.trim()) {
      return categoryStrategies;
    }

    const query = searchQuery.toLowerCase();
    return categoryStrategies.filter((strat) => {
      const name = typeof strat === 'string' ? strat : strat.name;
      const desc = getDescription(name);
      return name.toLowerCase().includes(query) || desc.toLowerCase().includes(query);
    });
  }, [categorizedStrategies, activeCategory, searchQuery, getDescription]);

  // Get actual counts for tabs (use category data or default counts)
  const getCategoryCount = useCallback((categoryKey) => {
    const strategies = categorizedStrategies[categoryKey];
    if (strategies && strategies.length > 0) {
      return strategies.length;
    }
    // Fall back to default counts
    const cat = STRATEGY_CATEGORIES.find((c) => c.key === categoryKey);
    return cat ? cat.count : 0;
  }, [categorizedStrategies]);

  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setActiveCategory(newValue);
    setSearchQuery(''); // Clear search when changing tabs
  };

  // Handle strategy selection
  const handleStrategyClick = (strategyName) => {
    if (onChange) {
      onChange(strategyName);
    }
  };

  // Handle search input
  const handleSearchChange = (event) => {
    setSearchQuery(event.target.value);
  };

  // Fetch description on hover or focus (lazy loading)
  const handleStrategyHover = (strategyName) => {
    if (!fetchedDescriptions[strategyName] && !loadingDescriptions[strategyName]) {
      fetchDescription(strategyName);
    }
  };

  return (
    <Box>
      {/* Category Tabs */}
      <Tabs
        value={activeCategory}
        onChange={handleTabChange}
        variant="scrollable"
        scrollButtons="auto"
        sx={{
          borderBottom: 1,
          borderColor: 'divider',
          mb: 2,
          '& .MuiTab-root': {
            minHeight: 48,
            textTransform: 'none',
          },
        }}
      >
        {STRATEGY_CATEGORIES.map((cat) => (
          <Tab
            key={cat.key}
            value={cat.key}
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <span>{cat.name}</span>
                <Chip
                  label={getCategoryCount(cat.key)}
                  size="small"
                  sx={{
                    height: 20,
                    fontSize: '0.7rem',
                    bgcolor: activeCategory === cat.key ? 'primary.main' : 'action.selected',
                    color: activeCategory === cat.key ? 'primary.contrastText' : 'text.secondary',
                  }}
                />
              </Box>
            }
          />
        ))}
      </Tabs>

      {/* Search Field */}
      <TextField
        fullWidth
        size="small"
        placeholder="Search strategies..."
        value={searchQuery}
        onChange={handleSearchChange}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon fontSize="small" color="action" />
            </InputAdornment>
          ),
        }}
        sx={{ mb: 2 }}
      />

      {/* Strategy List */}
      <Paper
        variant="outlined"
        sx={{
          maxHeight: 300,
          overflow: 'auto',
          '&::-webkit-scrollbar': {
            width: 8,
          },
          '&::-webkit-scrollbar-track': {
            bgcolor: 'action.hover',
            borderRadius: 4,
          },
          '&::-webkit-scrollbar-thumb': {
            bgcolor: 'action.selected',
            borderRadius: 4,
            '&:hover': {
              bgcolor: 'action.focus',
            },
          },
        }}
      >
        {loading ? (
          <Box sx={{ p: 2 }}>
            {[1, 2, 3, 4, 5].map((i) => (
              <Box key={i} sx={{ mb: 1.5 }}>
                <Skeleton height={24} width="60%" />
                <Skeleton height={16} width="80%" />
              </Box>
            ))}
          </Box>
        ) : filteredStrategies.length === 0 ? (
          <Box sx={{ p: 3, textAlign: 'center' }}>
            <Typography color="text.secondary">
              {searchQuery
                ? `No strategies matching "${searchQuery}"`
                : 'No strategies in this category'}
            </Typography>
          </Box>
        ) : (
          <List dense disablePadding>
            {filteredStrategies.map((strat) => {
              const name = typeof strat === 'string' ? strat : strat.name;
              const isSelected = value === name;
              const description = getDescription(name);

              return (
                <ListItem key={name} disablePadding>
                  <ListItemButton
                    selected={isSelected}
                    onClick={() => handleStrategyClick(name)}
                    onMouseEnter={() => handleStrategyHover(name)}
                    sx={{
                      py: 1.5,
                      px: 2,
                      bgcolor: isSelected ? 'action.selected' : 'transparent',
                      '&:hover': {
                        bgcolor: isSelected ? 'action.selected' : 'action.hover',
                      },
                      '&.Mui-selected': {
                        bgcolor: 'action.selected',
                        '&:hover': {
                          bgcolor: 'action.selected',
                        },
                      },
                    }}
                  >
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography
                            variant="body1"
                            sx={{
                              fontWeight: isSelected ? 600 : 400,
                              color: isSelected ? 'primary.main' : 'text.primary',
                            }}
                          >
                            {name}
                          </Typography>
                          {isSelected && (
                            <CheckCircleIcon
                              fontSize="small"
                              color="primary"
                            />
                          )}
                        </Box>
                      }
                      secondary={
                        <Typography
                          variant="body2"
                          color="text.secondary"
                          sx={{
                            mt: 0.5,
                            fontSize: '0.8rem',
                            lineHeight: 1.3,
                          }}
                        >
                          {loadingDescriptions[name] ? (
                            <Skeleton width="70%" />
                          ) : (
                            description
                          )}
                        </Typography>
                      }
                    />
                  </ListItemButton>
                </ListItem>
              );
            })}
          </List>
        )}
      </Paper>

      {/* Total count footer */}
      <Typography
        variant="caption"
        color="text.secondary"
        sx={{ display: 'block', mt: 1, textAlign: 'right' }}
      >
        {searchQuery
          ? `${filteredStrategies.length} results`
          : `${filteredStrategies.length} strategies in ${STRATEGY_CATEGORIES.find((c) => c.key === activeCategory)?.name || ''}`}
      </Typography>
    </Box>
  );
}
