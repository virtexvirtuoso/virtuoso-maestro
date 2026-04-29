# Academic Publication Strategy — Three Papers

Prepared by: Virtuoso Crypto Research
Date: 2026-02-13
Status: DRAFT — For Review

---

# PAPER 1

## Real-Time Monetary Liquidity Proxies for Cryptocurrency Position Management

### Authors
Fernando [Last Name] (Virtuoso Crypto), [Professor TBD], [Student TBD]

### Abstract

We construct a parsimonious real-time proxy for U.S. M2 money supply conditions using daily closing prices of four cross-asset instruments: the U.S. Dollar Index (DXY), gold (GLD), 10-year U.S. Treasury bonds (TLT), and high-yield corporate bonds (HYG). Our proxy employs a directional consensus mechanism — requiring at least three of four instruments to confirm monetary easing — to generate a daily-frequency liquidity signal that overcomes the approximately two-week publication lag inherent in Federal Reserve M2 data (FRED series M2SL). Applied to Bitcoin position management over the period 2018-2025, we find that BTC generates annualized returns of 102.8% during periods when our proxy signals monetary expansion versus 2.1% during contraction — a spread of 100.7 percentage points per year. In head-to-head comparison against a comprehensive Net Liquidity proxy (Federal Reserve balance sheet minus Treasury General Account minus Reverse Repo facility), our four-instrument proxy achieves comparable or superior risk-adjusted returns for cryptocurrency position timing despite requiring only four freely available daily market prices versus weekly Federal Reserve reporting data. Lead-lag analysis demonstrates that our proxy anticipates the direction of official M2 prints with [X]% accuracy at an average lead of [X] days. These findings suggest that cross-asset market prices embed monetary policy information in real time and that parsimonious proxy construction can outperform complex liquidity models for cryptocurrency applications. Our results are robust to walk-forward validation across 14 non-overlapping out-of-sample periods (Sharpe ratio 0.84, p = 0.036).

### Target Journals
1. Journal of Financial Economics
2. Journal of Monetary Economics
3. Journal of International Money and Finance

### Key Contribution
First systematic construction and validation of a real-time M2 proxy using cross-asset market prices for cryptocurrency applications. Demonstrates that four daily-frequency instruments subsume the information content of weekly Federal Reserve reporting data for crypto position timing.

---

# PAPER 2

## Do Traditional Equity Risk Factors Predict Cryptocurrency Market Regimes? Evidence from Cross-Domain Machine Learning

### Authors
Fernando [Last Name] (Virtuoso Crypto), [Professor TBD], [Student TBD], [Student TBD]

### Abstract

We investigate whether traditional equity risk factors carry predictive information for cryptocurrency market regime classification. Using a gradient-boosted decision tree classifier (LightGBM) trained on 111 engineered features spanning four distinct domains — macroeconomic indicators (Federal Reserve Economic Data), Fama-French five-factor plus momentum model outputs, cross-asset technical features, and cryptocurrency-native realized volatility measures — we classify daily cryptocurrency market conditions into five regimes: Bull, Mild Bull, Neutral, Bear, and Accumulation. Applying SHAP (SHapley Additive exPlanations) analysis, we find that traditional macroeconomic and academic equity factors dominate the feature importance ranking: M2 money supply acceleration ranks first, CPI year-over-year ranks second, and the Fama-French market excess return factor (Mkt-RF) ranks third — all ahead of crypto-native indicators. Most surprisingly, the profitability factor (RMW) from the Fama-French five-factor model ranks among the top five predictive features for cryptocurrency regime classification, a result with no precedent in the published literature. Walk-forward validation across 14 non-overlapping out-of-sample folds yields a Sharpe ratio of 1.33 for the ML-based regime classifier versus 0.61 for a rule-based alternative, with a Calmar ratio of 1.54. Ablation analysis confirms that removing the equity factor domain degrades out-of-sample performance below statistical significance, demonstrating that the cross-domain feature combination is non-separable. Our findings challenge the prevailing narrative that cryptocurrency markets constitute a distinct, uncorrelated asset class and suggest that traditional risk premia carry substantial information for crypto market timing.

### Target Journals
1. Review of Financial Studies
2. Journal of Financial and Quantitative Analysis
3. Journal of Financial Economics

### Key Contribution
First application of Fama-French five-factor model outputs as input features for supervised cryptocurrency regime classification. Discovery that RMW (profitability factor) predicts crypto regimes — a non-obvious, cross-domain finding connecting the equity factor zoo to digital asset markets.

---

# PAPER 3

## Adaptive Position Sizing via Multi-Signal Confluence: A Walk-Forward Validated Framework for Cryptocurrency Perpetual Futures

### Authors
Fernando [Last Name] (Virtuoso Crypto), [Professor TBD], [Student TBD]

### Abstract

We present a systematic framework for adaptive position sizing in cryptocurrency perpetual futures markets using a multi-signal confluence architecture. Five independent trading signals — M2 money supply acceleration, a real-time cross-asset liquidity proxy, yield curve dynamics, cross-asset momentum, and crypto-native trend following — are aggregated into an integer confluence score (0-5). These signals span four distinct data domains and exhibit an average pairwise correlation of 0.12, ensuring that each signal contributes genuine incremental information. The confluence score maps to a continuous leverage multiplier ranging from 0.3x (low conviction) to 2.0x (maximum conviction), with dynamic safety overrides including a volatility ceiling (leverage halved when 30-day realized volatility exceeds 80%) and a multi-level drawdown circuit breaker. Applied to a four-asset portfolio (BTC 40%, ETH 25%, SOL 20%, LINK 15%) over the period 2017-2025 with per-asset walk-forward-optimized parameters, the system achieves a Sharpe ratio of 1.71 with maximum drawdown of -5.5% on the long side with adaptive leverage. The short-side module generates +23.1% during the 2022 bear market while incurring zero losses during the COVID crash (March 2020), the May 2021 correction, and the FTX collapse (November 2022). Walk-forward validation across 14 non-overlapping out-of-sample folds yields a Sharpe ratio of 0.84 (p = 0.036). Ablation analysis demonstrates non-separability: removing any single signal degrades out-of-sample risk-adjusted returns, with M2 acceleration contributing the largest marginal effect. We compare the confluence approach against equal-weight signal combination and find that conviction-weighted leverage scaling produces superior risk-adjusted returns, particularly during regime transitions.

### Target Journals
1. Quantitative Finance
2. ICAIF (ACM International Conference on AI in Finance)
3. Journal of Portfolio Management

### Key Contribution
First walk-forward-validated multi-signal confluence framework for cryptocurrency position sizing that spans macro, cross-asset, academic factor, and crypto-native data domains. Demonstrates that low-correlation signal combination with conviction-weighted leverage scaling produces statistically significant out-of-sample alpha with controlled drawdowns.

---

# PROFESSOR OUTREACH EMAILS

## Template A — For Macro/Monetary Economics Faculty

Subject: Research Collaboration — Real-Time M2 Proxy for Crypto Markets

Dear Professor [Name],

I'm writing to explore a potential research collaboration on a topic at the intersection of monetary economics and cryptocurrency markets.

In my applied research, I've documented a striking empirical regularity: Bitcoin generates annualized returns of approximately 103% during periods of M2 money supply acceleration versus approximately 2% during deceleration — a spread that persists across multiple market cycles from 2017 to 2025.

To address the well-known publication lag in Federal Reserve M2 data (approximately two weeks at monthly frequency), I've constructed a parsimonious daily-frequency proxy using four cross-asset instruments (dollar index, gold, treasuries, and high-yield credit). Preliminary walk-forward validation suggests this proxy achieves comparable or superior performance to more complex liquidity models for cryptocurrency applications.

I believe this work raises interesting questions about how quickly monetary policy information is incorporated into cross-asset prices and whether simple market-based proxies can substitute for lagged official statistics — questions that extend well beyond cryptocurrency applications.

I would welcome the opportunity to discuss this further and explore whether a formal collaboration might be of mutual interest. I can provide the complete dataset, preliminary analysis, and working codebase.

Best regards,
Fernando
Virtuoso Crypto Research
[email redacted]

---

## Template B — For ML/Quantitative Finance Faculty

Subject: Research Collaboration — Fama-French Factors as Crypto Regime Predictors

Dear Professor [Name],

I'm reaching out regarding a finding from my applied quantitative research that I believe would be of interest to you, given your work on [reference their specific publication].

Using a LightGBM classifier trained on 111 features spanning macroeconomic indicators (FRED), Fama-French five-factor plus momentum outputs, cross-asset technical features, and cryptocurrency volatility measures, I've found that traditional equity risk factors — particularly the profitability factor (RMW) — rank among the top predictors of cryptocurrency market regimes. This result is robust to SHAP analysis and walk-forward validation across 14 out-of-sample folds.

The finding is surprising: it suggests that cryptocurrency markets, often characterized as a distinct and uncorrelated asset class, are in fact significantly informed by the same risk premia that drive equity returns. I believe this connects to the growing literature on crypto factor models (Liu, Tsyvinski, and Wu 2022) and challenges some prevailing assumptions about crypto market microstructure.

I'm preparing this work for formal publication and would greatly value your perspective. If a collaboration is of interest, I can offer co-authorship, access to the full dataset and analysis pipeline, and computational resources. I'm also open to involving graduate students who might find this a productive thesis topic.

Best regards,
Fernando
Virtuoso Crypto Research
[email redacted]

---

## Template C — For Stevens/NJIT/Baruch Faculty (More Practical Tone)

Subject: Guest Lecture + Student Research Opportunity — Quantitative Crypto Trading

Dear Professor [Name],

I lead the quantitative research team at Virtuoso Crypto, where we've built a walk-forward-validated systematic trading framework for cryptocurrency perpetual futures. Our system combines macroeconomic signals, Fama-French factor models, and machine learning regime classification — and we're preparing three papers for publication.

I'd like to propose two things:

1. A guest lecture for your [course name] students on applied quantitative finance in crypto markets — covering real-world challenges like walk-forward validation, overfitting, and the gap between backtests and live trading.

2. A research collaboration opportunity for 2-4 motivated students who want to contribute to publishable work. We have well-defined research tasks (ablation studies, robustness checks, alternative model specifications) that would make excellent thesis projects. Contributing students would receive co-authorship and a paid research stipend.

We're also running a quantitative finance challenge (the "UNDERTOW Challenge") that might be a good fit for your program — I'm happy to share details.

Would you be open to a brief call to discuss?

Best regards,
Fernando
Virtuoso Crypto Research
[email redacted]

---

# UNDERTOW CHALLENGE — CONCEPT BRIEF

## Overview
A multi-tier quantitative finance puzzle designed to identify talented students at Stevens Institute of Technology, NJIT, and Baruch College for research collaboration and potential internship opportunities.

## Challenge URL
undertowchallenge.com (to be built)

## Tier Structure

### Tier 1: "The Current" (Entry Level)
**Task:** Download FRED M2SL and BTC-USD daily prices (2017-2025). Compute rolling 3-month and 6-month M2 growth rates. Build a simple long/flat strategy: long BTC when 3m growth > 6m growth, flat otherwise. Report Sharpe ratio and maximum drawdown.

**Skills tested:** Data sourcing, pandas, basic backtesting
**Expected completion:** 2-4 hours
**Reward:** Access to Tier 2 + UNDERTOW community Discord

### Tier 2: "The Undertow" (Intermediate)
**Task:** Build a regime classifier for BTC using at least 3 data domains (macro, technical, and one other). Train with walk-forward validation (2-year train, 6-month test). Beat a Sharpe of 0.5 on out-of-sample data. Submit code + results.

**Skills tested:** ML, feature engineering, walk-forward methodology, avoiding overfitting
**Expected completion:** 1-2 weeks
**Reward:** Invitation to private research session + UNDERTOW swag

### Tier 3: "The Abyss" (Advanced)
**Task:** We claim Fama-French equity risk factors predict cryptocurrency market regimes. Replicate or disprove this claim. Provide SHAP analysis showing feature importance across walk-forward folds. If you can improve our classifier's OOS Sharpe of 1.33, show us how.

**Skills tested:** Advanced ML, academic rigor, original research
**Expected completion:** 2-4 weeks
**Reward:** Co-authorship opportunity + paid research internship ($2,000-5,000 stipend)

## Promotion Strategy
- QR code flyers near CS/finance/data science departments
- Cryptic messaging: "M2 accelerates. BTC follows. Prove it. undertowchallenge.com"
- Professor endorsement (after outreach)
- Reddit/Discord in university quant finance communities
- LinkedIn posts targeting Stevens/NJIT/Baruch students

## Budget
| Item | Cost |
|------|------|
| Domain + hosting | $50/year |
| Website build | 1-2 days (internal) |
| Flyer printing (200 copies x 3 campuses) | $300 |
| UNDERTOW swag (t-shirts, stickers) | $500 |
| Research stipends (3-5 students) | $10,000-$25,000 |
| **Total** | **$11,000-$26,000** |

---

*File provisional patent before any of this goes public.*
