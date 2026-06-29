# WTI Crude Oil Futures — Quantitative Backtesting Engine

**Author:** Kunal Firake | [GitHub](https://github.com/KunalFirake-sys) | [LinkedIn](https://www.linkedin.com/in/kunal-firake-46645128b/)

---

## Overview

A Python-based quantitative research pipeline that downloads 10 years of WTI crude oil futures (CL=F) price data, generates systematic trading signals, simulates portfolio performance, and critically evaluates results for overfitting, data-snooping bias, regime dependence, and transaction cost sensitivity.

The project follows a four-iteration research progression — each version motivated by findings from the previous one — culminating in a regime-switching strategy that combines a moving average crossover in bull markets with a Donchian channel breakout in bear markets.

**Instrument:** WTI Crude Oil Futures (CL=F, NYMEX)  
**Data range:** January 2015 – December 2024 (2,513 trading days)  
**Libraries:** pandas, numpy, scipy, statsmodels, matplotlib, seaborn, yfinance

---

## Strategy Progression

### V1 — Dual Moving Average Crossover with Momentum Filter
The baseline strategy. Generates a long signal when the 20-day MA crosses above the 50-day MA and momentum (10-day ROC) confirms, and a short signal on the reverse. No regime awareness.

### V2 — Bull Regime Filter (V1 + 200-day MA Gate)
Motivated by regime analysis showing V1 only has edge in bull markets (Sharpe +0.42). Adds a 200-day MA regime filter: long signals only fire when price is above the 200-day MA, short signals only when below. Strategy sits flat when regime is ambiguous.

### V3 — Donchian Channel Breakout with Regime Filter
A separate signal approach. Buys when price breaks above the 20-day highest high (confirmed by bull regime), sells short when price breaks below the 20-day lowest low (confirmed by bear regime). Tested independently to compare signal quality against the MA crossover approach.

### V4 — Regime-Switching Combined Strategy (Final)
Combines V2 and V3 into a single regime-aware framework:
- **Bull regime** (price > 200-day MA): use MA crossover signal — long only
- **Bear regime** (price < 200-day MA): use breakout signal — short only
- Flat otherwise

This ensures the strategy always has an appropriate signal for prevailing market conditions rather than sitting idle during bear periods.

---

## Performance Summary

> Note: Sharpe ratios appear extreme due to very low portfolio volatility (0.32–0.50% annualised) caused by high cash allocation during flat periods. Total return, max drawdown, win rate, and profit factor are the meaningful metrics here.

| Metric | V1 (MA Cross) | V2 (Regime Filter) | V3 (Breakout) | V4 (Combined) |
|--------|--------------|-------------------|--------------|--------------|
| Total Return | -0.53% | -0.72% | -1.85% | **-0.45%** |
| CAGR | -0.05%/yr | -0.07%/yr | -0.19%/yr | **-0.05%/yr** |
| Max Drawdown | -1.61% | -1.72% | -2.05% | **-1.54%** |
| Win Rate | 40.71% | 39.58% | 47.85% | **48.84%** |
| Profit Factor | 0.90 | 0.84 | 0.56 | **0.91** |
| Total Trades | 226 | 192 | 326 | 257 |
| Avg Trade Duration | 15.6 days | 17.5 days | 10.0 days | 12.7 days |

V4 achieves the best result across every meaningful metric — lowest loss, lowest drawdown, highest win rate, and highest profit factor.

---

## Critical Analysis

### 1. Overfitting — In-Sample vs Out-of-Sample

Data split at 31 December 2021:
- **In-Sample (2015–2021):** Return -0.04%, consistent performance
- **Out-of-Sample (2022–2024):** Return -0.98%, slight degradation

A scipy t-test on daily returns across both periods produced p = 0.334, indicating no statistically significant difference. The strategy performs consistently across seen and unseen data — confirming the result is structural rather than overfitted to historical noise.

### 2. Data-Snooping Bias — Parameter Sensitivity

A 5×5 grid of 25 MA window combinations (fast: 10–30 days, slow: 40–80 days) was tested. All 25 combinations produced negative Sharpe ratios ranging from -8.76 to -10.37, uniformly distributed across the grid. This rules out parameter-specific luck — the MA crossover signal is fundamentally mismatched to crude oil's price behaviour regardless of window choice.

Notably, when the regime filter (V2 logic) was applied during sensitivity testing, four combinations produced positive total returns, suggesting the regime filter is the critical component unlocking whatever edge exists in the signal.

### 3. Regime Dependence

Performance varies significantly by market condition:

| Regime | Sharpe | Ann. Return |
|--------|--------|-------------|
| Bull (price > 200d MA) | **+0.60** | **+0.29%** |
| Bear (price < 200d MA) | -0.97 | -0.44% |
| High Volatility | -0.21 | -0.09% |
| Low Volatility | **+0.08** | **+0.03%** |

The strategy has genuine edge exclusively in bull market conditions. This finding directly motivated V4's design — deploying a different signal logic in bear regimes rather than sitting idle.

### 4. Transaction Cost Sensitivity

Backtest run across costs from 0% to 0.50% per trade. The Sharpe ratio remains in the -10 to -12 range across all cost levels, confirming transaction costs are not the primary driver of underperformance. The signal itself is the issue — not the fee structure. This is an important negative result: reducing costs alone cannot rescue a structurally weak signal.

---

## Key Findings

1. **MA crossover signals do not suit crude oil's price dynamics.** Crude oil's sensitivity to exogenous shocks (OPEC decisions, geopolitical events, inventory reports) creates frequent sharp reversals that whipsaw crossover signals regardless of parameter choice.

2. **Regime filtering is the critical variable.** Adding a 200-day MA regime gate reduces total losses by 94% (V1: -12.67% without min_hold → V2: -0.72%) and improves bull-regime Sharpe from +0.42 to +0.54.

3. **Combining signals by regime outperforms either signal alone.** V4 achieves the best win rate (48.84%) and profit factor (0.91) by using the most appropriate signal type for each market condition.

4. **The strategy successfully avoided the April 2020 crude oil price collapse** (WTI went negative for the first time in history), preserving capital while buy-and-hold portfolios lost over 170% of initial capital temporarily.

5. **No overfitting detected across any strategy version.** IS/OOS performance differences are not statistically significant (p > 0.30 for all versions).

---

## Limitations

- All strategies remain unprofitable in absolute terms over the test period, underperforming a simple buy-and-hold on crude oil (+34.73%)
- Daily bar data misses intraday price dynamics that are central to real futures trading
- Fixed fractional position sizing does not account for futures contract roll costs or margin requirements
- The 200-day MA regime definition is a simplification — real regime detection would use more sophisticated methods (Hidden Markov Models, volatility clustering)
- Sharpe ratio is not an appropriate metric for low-frequency strategies with high cash allocation; not used as primary performance metric
- No walk-forward optimisation performed — parameters are fixed across the full period

---

## Future Research

- Walk-forward optimisation to avoid look-ahead bias in parameter selection
- Incorporate EIA weekly inventory report data as a fundamental signal layer
- Test RSI mean-reversion as an alternative signal for range-bound (sideways) regimes
- Apply Hidden Markov Models for more robust regime detection
- Extend to other energy futures (Brent crude, natural gas) to test signal generalisability

---

## Project Structure

```
crude_oil_quant/
│
├── data/
│   └── cl_futures_data.csv        # Raw OHLCV data from Yahoo Finance
│
├── src/
│   ├── data_loader.py             # Downloads and cleans CL=F data
│   ├── signals.py                 # V1, V2, V3, V4 signal generators
│   ├── backtest.py                # Portfolio simulation engine
│   ├── metrics.py                 # Performance metric calculations
│   ├── analysis.py                # IS/OOS split, parameter grid, regime analysis
│   └── visualise.py               # All chart generation
│
├── notebooks/
│   └── full_analysis.ipynb        # End-to-end analysis notebook
│
├── results/
│   ├── equity_curve_all_strategies.png
│   ├── parameter_heatmap.png
│   ├── regime_performance.png
│   ├── is_oos_comparison.png
│   ├── cost_sensitivity.png
│   ├── drawdown.png
│   ├── signals.png
│   └── v3_signals.png
│
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/KunalFirake-sys/crude_oil_quant.git
cd crude_oil_quant

# Install dependencies
pip install -r requirements.txt

# Run a specific strategy version (V1, V2, V3, or V4)
python main.py --strategy V4

# Or open the full analysis notebook
jupyter notebook notebooks/full_analysis.ipynb
```

---

## Dependencies

```
yfinance>=0.2.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
statsmodels>=0.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

---

*Built as a portfolio project demonstrating quantitative research methodology: systematic signal generation, rigorous backtesting, and critical evaluation of results for overfitting, data-snooping bias, and regime dependence.*
