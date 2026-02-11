"""Ichimoku Cloud Strategy"""
import pandas as pd
import numpy as np

NAME = "Ichimoku"
CATEGORY = "technical"
DESCRIPTION = "Ichimoku Cloud breakout with trend confirmation"
REQUIRES_DERIVATIVES = False

def generate_signals(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> pd.Series:
    """Ichimoku Cloud signals."""
    signals = pd.Series(0, index=df.index)
    
    # Tenkan-sen (Conversion Line)
    tenkan_high = df['high'].rolling(tenkan).max()
    tenkan_low = df['low'].rolling(tenkan).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line)
    kijun_high = df['high'].rolling(kijun).max()
    kijun_low = df['low'].rolling(kijun).min()
    kijun_sen = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A)
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    
    # Senkou Span B (Leading Span B)
    senkou_b_high = df['high'].rolling(senkou_b).max()
    senkou_b_low = df['low'].rolling(senkou_b).min()
    senkou_span_b = ((senkou_b_high + senkou_b_low) / 2).shift(kijun)
    
    # Cloud top and bottom
    cloud_top = pd.concat([senkou_a, senkou_span_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([senkou_a, senkou_span_b], axis=1).min(axis=1)
    
    # Signals: Price above cloud = long, below = short
    signals[df['close'] > cloud_top] = 1
    signals[df['close'] < cloud_bottom] = -1
    
    return signals
