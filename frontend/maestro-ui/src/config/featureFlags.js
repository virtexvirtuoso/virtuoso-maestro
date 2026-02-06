// featureFlags.js - Feature flags for gradual migrations
// Controls which chart library to use (Highcharts vs TradingView)

export const FEATURES = {
  // When true, use TradingView Lightweight Charts instead of Highcharts
  // Set REACT_APP_USE_TRADINGVIEW=true in .env.local to enable
  USE_TRADINGVIEW: process.env.REACT_APP_USE_TRADINGVIEW === 'true',
};

export default FEATURES;
