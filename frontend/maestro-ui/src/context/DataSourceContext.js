// DataSourceContext - Central data source management with caching
import React, { createContext, useContext, useState, useEffect, useCallback, useMemo } from 'react';

const DataSourceContext = createContext(null);

// Provider symbol counts from maestro-dev.yaml config
const PROVIDER_CONFIG = {
  binance: {
    name: 'Binance',
    description: 'Largest crypto exchange by volume',
    color: '#F0B90B',
    symbols: ['btcusdt', 'ethusdt', 'solusdt', 'cgptusdt', 'zecusdt', 'sentusdt', 'cocosusdt',
              'renderusdt', 'fetusdt', 'taousdt', 'arbusdt', 'opusdt', 'suiusdt', 'tiausdt',
              'injusdt', 'linkusdt', 'avaxusdt', 'ethbtc', 'ltcbtc', 'zecbtc', 'chzbtc', 'iostbtc'],
  },
  bitmex: {
    name: 'BitMEX',
    description: 'Pioneer of perpetual contracts',
    color: '#5B98D8',
    symbols: ['xbtusd'],
  },
  bybit: {
    name: 'Bybit',
    description: 'Derivatives and spot trading',
    color: '#F7A600',
    symbols: ['myxusdt', 'riverusdt'],
  },
  mexc: {
    name: 'MEXC',
    description: 'Wide altcoin selection',
    color: '#2CA6A4',
    symbols: ['myxusdt', 'rainusdt', 'dstusdt', 'riverusdt', 'cocousdt'],
  },
  kucoin: {
    name: 'KuCoin',
    description: 'Major altcoin exchange',
    color: '#23AF91',
    symbols: ['rainusdt', 'adiusdt'],
  },
  gate: {
    name: 'Gate.io',
    description: 'Spot and margin trading',
    color: '#17E6A1',
    symbols: ['myxusdt'],
  },
};

// Strategy categories from backend/strategies/__init__.py
const STRATEGY_CATEGORIES = {
  technical: { name: 'Technical', count: 19, description: 'Pure TA indicators' },
  scalping: { name: 'Scalping', count: 8, description: 'Short-term momentum' },
  momentum: { name: 'Momentum', count: 6, description: 'Trend-following' },
  composite: { name: 'Composite', count: 6, description: 'Multi-indicator combos' },
  derivatives: { name: 'Derivatives', count: 11, description: 'OI, CVD, funding' },
  hybrids: { name: 'Hybrids', count: 15, description: 'Volume + trend filters' },
};

export function DataSourceProvider({ children }) {
  const [providers, setProviders] = useState([]);
  const [providerSymbols, setProviderSymbols] = useState({});
  const [strategies, setStrategies] = useState([]);
  const [strategyCategories, setStrategyCategories] = useState({});
  const [recentOptimizations, setRecentOptimizations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch available providers
  const fetchProviders = useCallback(async () => {
    try {
      const response = await fetch(`${process.env.REACT_APP_REST_API_URL}/datasource/available`);
      const data = await response.json();
      setProviders(data);

      // Fetch symbols for each provider
      const symbolPromises = data.map(async (provider) => {
        try {
          const symResponse = await fetch(`${process.env.REACT_APP_REST_API_URL}/datasource/${provider}/symbols`);
          const symbols = await symResponse.json();
          return { provider, symbols };
        } catch {
          return { provider, symbols: PROVIDER_CONFIG[provider]?.symbols || [] };
        }
      });

      const results = await Promise.all(symbolPromises);
      const symbolMap = {};
      results.forEach(({ provider, symbols }) => {
        symbolMap[provider] = symbols;
      });
      setProviderSymbols(symbolMap);
    } catch (e) {
      // Fallback to config
      setProviders(Object.keys(PROVIDER_CONFIG));
      const fallbackSymbols = {};
      Object.entries(PROVIDER_CONFIG).forEach(([key, config]) => {
        fallbackSymbols[key] = config.symbols;
      });
      setProviderSymbols(fallbackSymbols);
    }
  }, []);

  // Fetch available strategies
  const fetchStrategies = useCallback(async () => {
    try {
      const response = await fetch(`${process.env.REACT_APP_REST_API_URL}/strategy/available`);
      const data = await response.json();
      setStrategies(data);

      // Try to get category counts
      try {
        const catResponse = await fetch(`${process.env.REACT_APP_REST_API_URL}/strategy/categories`);
        const catData = await catResponse.json();
        setStrategyCategories(catData);
      } catch {
        setStrategyCategories(STRATEGY_CATEGORIES);
      }
    } catch {
      // Fallback: use static count
      setStrategies(Array(65).fill('strategy'));
      setStrategyCategories(STRATEGY_CATEGORIES);
    }
  }, []);

  // Fetch recent optimizations
  const fetchRecentOptimizations = useCallback(async () => {
    try {
      const response = await fetch(`${process.env.REACT_APP_REST_API_URL}/optimization/recent?limit=5`);
      if (response.ok) {
        const data = await response.json();
        setRecentOptimizations(data);
      }
    } catch {
      // No recent optimizations available
      setRecentOptimizations([]);
    }
  }, []);

  // Initial data load
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      setError(null);
      try {
        await Promise.all([
          fetchProviders(),
          fetchStrategies(),
          fetchRecentOptimizations(),
        ]);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, [fetchProviders, fetchStrategies, fetchRecentOptimizations]);

  // Computed stats
  const stats = useMemo(() => {
    const totalSymbols = Object.values(providerSymbols).reduce(
      (sum, symbols) => sum + symbols.length,
      0
    );
    const totalStrategies = strategies.length || 65;

    return {
      providerCount: providers.length || 6,
      symbolCount: totalSymbols || 31,
      strategyCount: totalStrategies,
      optimizationCount: recentOptimizations.length,
    };
  }, [providers, providerSymbols, strategies, recentOptimizations]);

  // Get provider info with enriched data
  const getProviderInfo = useCallback((providerKey) => {
    const config = PROVIDER_CONFIG[providerKey] || {};
    const symbols = providerSymbols[providerKey] || config.symbols || [];
    return {
      key: providerKey,
      name: config.name || providerKey,
      description: config.description || '',
      color: config.color || '#666',
      symbols,
      symbolCount: symbols.length,
    };
  }, [providerSymbols]);

  // Get all providers with enriched info
  const enrichedProviders = useMemo(() => {
    return providers.map(getProviderInfo);
  }, [providers, getProviderInfo]);

  const value = {
    providers,
    providerSymbols,
    enrichedProviders,
    strategies,
    strategyCategories,
    recentOptimizations,
    stats,
    loading,
    error,
    getProviderInfo,
    refreshData: () => {
      fetchProviders();
      fetchStrategies();
      fetchRecentOptimizations();
    },
  };

  return (
    <DataSourceContext.Provider value={value}>
      {children}
    </DataSourceContext.Provider>
  );
}

export function useDataSources() {
  const context = useContext(DataSourceContext);
  if (!context) {
    throw new Error('useDataSources must be used within a DataSourceProvider');
  }
  return context;
}
