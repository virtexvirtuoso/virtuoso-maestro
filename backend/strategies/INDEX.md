# Maestro Strategy Library Index

**65 trading strategies** organized by category with entry/exit logic and mathematical formulas.

## Quick Reference

| Category | Count | Timeframe | Description |
|----------|-------|-----------|-------------|
| [Technical](#technical-19-strategies) | 19 | 15m-4h | Classic TA indicators |
| [Scalping](#scalping-8-strategies) | 8 | 1m-15m | High-frequency, tight targets |
| [Momentum](#momentum-6-strategies) | 6 | 1h-1d | Trend-following and mean reversion |
| [Composite](#composite-6-strategies) | 6 | 5m-1h | Multi-indicator confluence |
| [Derivatives](#derivatives-11-strategies) | 11 | 1h-4h | Funding, OI, liquidations |
| [Hybrids](#hybrids-15-strategies) | 15 | 5m-4h | Base strategy + filter combinations |

---

## Mathematical Notation

| Symbol | Formula |
|--------|---------|
| `EMA(n)` | `(close × α) + (EMA_prev × (1-α))` where `α = 2/(n+1)` |
| `SMA(n)` | `Σclose / n` |
| `RSI(n)` | `100 - (100 / (1 + avg_gain/avg_loss))` |
| `ATR(n)` | `SMA(max(H-L, |H-C_prev|, |L-C_prev|), n)` |
| `BB(n,σ)` | `SMA ± (σ × StdDev)` |
| `MACD` | `EMA(12) - EMA(26)`, Signal = `EMA(MACD, 9)` |
| `Z-score` | `(value - mean) / std_dev` |
| `CVD` | `Σ(volume × (2 × close_position - 1))` |

---

## Technical (19 strategies)

| Strategy | File | Logic | Indicators |
|----------|------|-------|------------|
| **ADXMomentum** | `adx_momentum.py` | ADX > 25 + DI confirmation | ADX, +DI, -DI |
| **ADXSmas** | `adx_smas.py` | ADX confirms trend + SMA cross | ADX, SMA(10/30) |
| **AsianBreakout** | `asian_breakout.py` | Break of Asian session range | Session levels |
| **AwesomeMACD** | `awesome_macd.py` | MACD + Awesome Oscillator agree | MACD, AO |
| **BBandRSI** | `bband_rsi.py` | Lower BB + RSI < 30 | BB(20,2), RSI(14) |
| **BollingerBreakout** | `bollinger_breakout.py` | Price breaks upper/lower band | BB(20,2) |
| **CCICMO** | `cci_cmo.py` | CCI + CMO confluence | CCI(20), CMO(14) |
| **Channel** | `channel.py` | Donchian Channel breakout | Donchian(20) |
| **EMA_Cross** | `ema_cross.py` | Fast/slow EMA crossover | EMA(12/26) |
| **Ichimoku** | `ichimoku.py` | Price vs cloud + TK cross | Tenkan, Kijun, Senkou |
| **MACD** | `macd.py` | MACD crosses signal | MACD(12,26,9) |
| **MeanReversion** | `mean_reversion.py` | Z-score < -2 or > 2 | Z-score, SMA(20) |
| **Momentum** | `momentum.py` | 14-period momentum direction | ROC(14) |
| **MultiRSI** | `multi_rsi.py` | All RSI periods agree | RSI(7/14/21) |
| **OBV** | `obv.py` | OBV vs OBV_SMA | OBV, SMA(20) |
| **RSI** | `rsi.py` | RSI < 30 or > 70 | RSI(14) |
| **SessionMomentum** | `session_momentum.py` | London/NY open momentum | Momentum, Sessions |
| **SMA_Cross** | `sma_cross.py` | SMA fast/slow crossover | SMA(10/30) |
| **VolumeBreakout** | `volume_breakout.py` | 2x volume + price direction | Volume, SMA |

### Entry Formulas

```python
# ADXMomentum
long = (ADX > 25) & (plus_di > minus_di)
short = (ADX > 25) & (minus_di > plus_di)

# BollingerBreakout
long = close > (SMA + 2σ)
short = close < (SMA - 2σ)

# MACD
long = (MACD > Signal) & (MACD.shift(1) <= Signal.shift(1))
short = (MACD < Signal) & (MACD.shift(1) >= Signal.shift(1))

# RSI
long = RSI < 30
short = RSI > 70

# MeanReversion
Z = (close - SMA) / StdDev
long = Z < -2.0
short = Z > 2.0
```

---

## Scalping (8 strategies)

| Strategy | File | Logic | Indicators |
|----------|------|-------|------------|
| **EMARibbon** | `ema_ribbon.py` | All EMAs aligned | EMA(8/13/21/34/55) |
| **GridTrading** | `grid_trading.py` | Buy bottom 20%, sell top 20% | Range detection |
| **MomentumBreakout** | `momentum_breakout.py` | Momentum > 2% + volume | Momentum, Volume |
| **Quickie** | `quickie.py` | Fast Z-score reversion | Z-score(10) |
| **ScalpRSI** | `scalp_rsi.py` | RSI(7) extremes | RSI(7) |
| **SmoothScalp** | `smooth_scalp.py` | EMA cross + RSI filter | EMA(5/13), RSI |
| **StochRSI** | `stoch_rsi.py` | StochRSI K/D cross | StochRSI(14) |
| **VWAP** | `vwap.py` | Price vs VWAP ± 2σ | VWAP, StdDev |

### Entry Formulas

```python
# EMARibbon
long = (EMA8 > EMA13) & (EMA13 > EMA21) & (EMA21 > EMA34) & (EMA34 > EMA55)
short = (EMA8 < EMA13) & (EMA13 < EMA21) & (EMA21 < EMA34) & (EMA34 < EMA55)

# VWAP
VWAP = Σ(typical_price × volume) / Σvolume
long = close < (VWAP - 2 × StdDev)
short = close > (VWAP + 2 × StdDev)

# StochRSI
K = 100 × (RSI - RSI_min) / (RSI_max - RSI_min)
long = (K < 20) & (K > D) & (K > K.shift(1))
short = (K > 80) & (K < D) & (K < K.shift(1))
```

---

## Momentum (6 strategies)

| Strategy | File | Logic | Indicators |
|----------|------|-------|------------|
| **BasisTrading** | `basis_trading.py` | Funding Z-score extremes | Funding, SMA, StdDev |
| **LiquidationHunt** | `liquidation_hunt.py` | Fade liquidation cascades | Volume(3x), Price |
| **MeanReversionBands** | `mean_reversion_bands.py` | Keltner Channel extremes | EMA, ATR |
| **TrendFollowingATR** | `trend_following_atr.py` | ATR trailing + SMA trend | ATR, SMA(50) |
| **TSMOM** | `tsmom.py` | Vol-scaled momentum | Momentum, Volatility |
| **VolatilityBreakout** | `volatility_breakout.py` | ATR expansion + direction | ATR, Momentum |

### Entry Formulas

```python
# TSMOM (Time-Series Momentum)
returns = pct_change()
vol = std(returns, 20) × √252
momentum = pct_change(20)
signal = sign(momentum) × (vol_target / vol)
long = signal > 0.5
short = signal < -0.5

# VolatilityBreakout
TR = max(H-L, |H-C_prev|, |L-C_prev|)
ATR = SMA(TR, 14)
vol_expanding = ATR > (ATR_avg × 1.5)
momentum = pct_change(5)
long = vol_expanding & (momentum > 0)
short = vol_expanding & (momentum < 0)

# Keltner Channels (MeanReversionBands)
upper = EMA(20) + 2 × ATR(14)
lower = EMA(20) - 2 × ATR(14)
long = close < lower
short = close > upper
```

---

## Composite (6 strategies)

| Strategy | File | Logic | Indicators |
|----------|------|-------|------------|
| **CombinedBinCluc** | `combined_bin_cluc.py` | BB + RSI OR EMA + volume | BB, RSI, EMA, Volume |
| **EMASkipPump** | `ema_skip_pump.py` | EMA cross, skip >5% moves | EMA(12/26), Price |
| **Fernando** | `fernando.py` | BBW + VLI (ETH Zurich thesis) | BBW, VLI, Volume |
| **LowBB** | `low_bb.py` | Mean reversion at lower BB | BB(20,2) |
| **ReinforcedAverage** | `reinforced_average.py` | All SMAs aligned | SMA(5/10/20/50) |
| **SmoothOperator** | `smooth_operator.py` | EMA + RSI + MACD agree | EMA, RSI, MACD |

### Entry Formulas

```python
# Fernando (BBW + VLI)
BBW = (upper - lower) / middle
VLI_fast = SMA(BBW, 20)
VLI_slow = SMA(BBW, 100)
vol_low = BBW < VLI_slow
long = vol_low & (close < lower) & volume_surge

# LowBB (Mean Reversion)
long = close <= (SMA - 2σ)
short = close >= (SMA + 2σ)

# ReinforcedAverage
long = (SMA5 > SMA10) & (SMA10 > SMA20) & (SMA20 > SMA50) & (close > SMA5)
```

---

## Derivatives (11 strategies)

*Require real derivatives data or use proxy calculations*

| Strategy | File | Logic | Indicators |
|----------|------|-------|------------|
| **CrossExchangeOI** | `cross_exchange_oi.py` | OI divergence vs price | OI, Price |
| **CVDScalp** | `cvd_scalp.py` | CVD divergence from price | CVD |
| **DeltaFlow** | `delta_flow.py` | Order flow delta extremes | Buy/Sell Delta |
| **FundingRate** | `funding_rate.py` | Fade extreme funding | Funding Rate |
| **LiquidationCascade** | `liquidation_cascade.py` | Post-cascade reversal | Volume, Wicks, OI |
| **LiquidationScalp** | `liquidation_scalp.py` | Quick liq reversals | Liquidation Data |
| **MicroBasis** | `micro_basis.py` | Micro TF funding scalp | Funding Rate |
| **RealFundingRate** | `real_funding_rate.py` | Funding Z-score | Funding, Mean, StdDev |
| **RealOIMomentum** | `real_oi_momentum.py` | OI + price confirmation | OI, Price |
| **SpotPerpBasis** | `spot_perp_basis.py` | Cumulative funding Z-score | Cum Funding |
| **VolRegimeFunding** | `vol_regime_funding.py` | Vol regime + funding | ATR, Funding |

### Entry Formulas

```python
# CVD (Cumulative Volume Delta)
close_position = (close - low) / (high - low)
delta_ratio = 2 × close_position - 1
volume_delta = delta_ratio × volume
CVD = cumsum(volume_delta)
bullish_div = (price_lower_low) & (CVD_higher_low)
bearish_div = (price_higher_high) & (CVD_lower_high)

# FundingRate
funding_ma = SMA(funding, 168)  # 7 days
funding_std = StdDev(funding, 168)
funding_zscore = (funding - funding_ma) / funding_std
long = funding_zscore < -2  # shorts crowded
short = funding_zscore > 2  # longs crowded

# LiquidationCascade
sharp_move = |close - open| > (ATR × 3)
volume_spike = volume > (vol_ma × 3)
long_wick = lower_wick / range > 0.6
long = sharp_down & volume_spike & long_wick  # fade
```

---

## Hybrids (15 strategies)

*Base strategy + filter for higher quality signals*

| Strategy | File | Base | Filter |
|----------|------|------|--------|
| **BollingerBreakout+CapitulationFilter** | `bollinger_capitulation.py` | BB | Capitulation/Euphoria ± 3% |
| **BollingerBreakout+OBV** | `bollinger_obv.py` | BB | OBV > OBV_SMA |
| **CapitulationCascade** | `capitulation_cascade.py` | Capitulation/Euphoria | OI + RSI extremes |
| **CapitulationReversal+VolumeFilter** | `capitulation_reversal_volume.py` | Volume | ±3% price move |
| **DerivativesCombo** | `derivatives_combo.py` | Voting | 2/3 of OI, CVD, Funding |
| **EMA_Cross+VolumeFilter** | `ema_cross_volume.py` | EMA | 2x volume |
| **MACD+RSI** | `macd_rsi.py` | MACD | RSI agreement |
| **MACD+VolumeFilter** | `macd_volume.py` | MACD | 2x volume |
| **MeanReversion+CapitulationFilter** | `mean_reversion_capitulation.py` | Z-score | Capitulation/Euphoria |
| **Momentum+TrendFilter** | `momentum_trend.py` | Momentum | SMA(50) trend |
| **OBV+TrendFilter** | `obv_trend.py` | OBV | SMA(50) trend |
| **OIDivergence+FundingRate** | `oi_funding.py` | OI div | Funding agree |
| **RSI+CapitulationFilter** | `rsi_capitulation.py` | RSI | Capitulation/Euphoria |
| **TripleConfirmation** | `triple_confirmation.py` | MACD | RSI + Volume |
| **VolumeBreakout+TrendFilter** | `volume_breakout_trend.py` | Volume | SMA(50) trend |

### Entry Formulas

```python
# Capitulation/Euphoria Filter (used by 4 hybrids)
vol_spike = volume > (vol_avg × 4)
price_drop = pct_change() < -0.03
price_pump = pct_change() > 0.03
capitulation = vol_spike & price_drop  # panic selling → fade long
euphoria = vol_spike & price_pump      # FOMO buying → fade short

# RSI+CapitulationFilter
long = capitulation & (RSI < 30)
short = euphoria & (RSI > 70)

# MeanReversion+CapitulationFilter
long = capitulation & (Z < -2)
short = euphoria & (Z > 2)

# BollingerBreakout+CapitulationFilter
long = capitulation & (close < lower_band)
short = euphoria & (close > upper_band)

# TripleConfirmation
macd_bull = MACD > Signal
rsi_oversold = RSI < 30
volume_high = volume > (vol_avg × 2)
long = macd_bull & rsi_oversold & volume_high

# DerivativesCombo (Voting)
oi_signal = (OI_up & price_down) ? 1 : (OI_up & price_up) ? -1 : 0
cvd_signal = (CVD_up & price_down) ? 1 : (CVD_down & price_up) ? -1 : 0
funding_signal = (funding < -0.0001) ? 1 : (funding > 0.0005) ? -1 : 0
votes = oi_signal + cvd_signal + funding_signal
long = votes >= 2
short = votes <= -2
```

---

## Strategy Selection Guide

| Goal | Recommended Strategies |
|------|----------------------|
| **Trend Following** | TSMOM, TrendFollowingATR, EMARibbon, Ichimoku |
| **Mean Reversion** | MeanReversion, LowBB, VWAP, BBandRSI |
| **Breakout** | BollingerBreakout, Channel, VolumeBreakout, VolatilityBreakout |
| **Scalping** | ScalpRSI, Quickie, StochRSI, CVDScalp |
| **Derivatives** | FundingRate, RealOIMomentum, LiquidationCascade |
| **High Conviction** | TripleConfirmation, DerivativesCombo, Fernando |

---

## Usage

```python
from strategies import list_strategies, run_strategy, summary

# List all strategies
print(list_strategies())  # 65 strategies

# List by category
print(list_strategies('derivatives'))  # 11 strategies

# Get summary
print(summary())
# {'technical': 19, 'scalping': 8, 'momentum': 6,
#  'composite': 6, 'derivatives': 11, 'hybrids': 15, 'total': 65}

# Run a strategy
signals = run_strategy('Fernando', df)
signals = run_strategy('MACD+RSI', df)
```

---

*Maestro Strategy Library - 65 bidirectional strategies (long + short) across 6 categories*
