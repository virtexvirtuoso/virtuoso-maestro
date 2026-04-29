from enum import Enum


class DataSourceProviders(Enum):
    BITMEX = 'BITMEX'
    BINANCE = 'BINANCE'
    BYBIT = 'BYBIT'
    OKX = 'OKX'
    GATE = 'GATE'
    KUCOIN = 'KUCOIN'
    MEXC = 'MEXC'
    YFINANCE = 'YFINANCE'
    FRED = 'FRED'
    FAMA_FRENCH = 'FAMA_FRENCH'
    ALPHA_VANTAGE = 'ALPHA_VANTAGE'
