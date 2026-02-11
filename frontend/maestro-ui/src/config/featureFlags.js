// featureFlags.js - Feature flags for gradual migrations
// Controls which chart library to use (Highcharts vs TradingView)

export const FEATURES = {
  // When true, use TradingView Lightweight Charts instead of Highcharts
  // Now enabled by default for dark theme trading experience
  USE_TRADINGVIEW: process.env.REACT_APP_USE_TRADINGVIEW !== 'false',
};

export default FEATURES;
