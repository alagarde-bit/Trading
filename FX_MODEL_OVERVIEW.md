# FX Trading Model – Overview for External Review

**Purpose:** Shareable summary of the current GBPAUD prediction/trading pipeline and latest backtest results, for use with Perplexity or other tools to get targeted advice.

---

## 1. Goal

Predict **5-day forward return** for GBPAUD and use it for **directional trading**: only take trades when predicted move is large enough, with stop-loss and confidence-based sizing. Target better **Sharpe and drawdown**, not raw return.

---

## 2. Data

- **Price/OHLC:** Daily (and aggregated weekly/monthly) from `combined_GBPAUD_daily.csv` (~5.1k rows after COT filter).
- **COT (CFTC):** `TFF_-_Futures_Only_*.csv` merged by report date (backward fill). GBP and AUD futures; Dealer / Asset Manager / Leveraged Money long–short nets, OI, and derived series.
- **Training filter:** `TRAIN_COT_ONLY = True` → only rows with valid COT for both currencies (~5,139 rows, 92 dropped).

---

## 3. Features (stationary / relative)

All inputs are **relative or normalized** (no raw price levels):

- **Returns & price structure:** Daily/weekly/monthly returns, range%, body%, close vs SMAs (20/50/200 daily; 4/10/40 weekly; 1/3/10 monthly), SMA ratios, cross-timeframe close ratios.
- **Volatility:** ATR%, BB width, 5d/20d/22d realized vol, vol ratio, first differences and HAR-style 5d/22d averages.
- **Momentum:** ROC 5/20, MACD (line/signal/hist), RSI (daily/weekly/monthly), first and second differences (e.g. RSI speed/acceleration).
- **Persistence / area:** Rolling sums of (close−SMA)/SMA over 10/20 days; rolling sums and counts of extreme COT z-scores over 4/8/12 weeks.
- **Distribution / autocorrelation:** 10d/20d autocorr of returns and squared returns, skewness, kurtosis.
- **COT:** Net positions and z-scores (GBP/AUD, Dealer/Asset_Mgr/Lev_Money), GBP−AUD diff, Lev−AssetMgr spread per currency and GBP−AUD of that, 4w/8w changes, 52w percentile, position-pain (z × 5d/20d return).
- **Patterns:** Candlestick and chart patterns (simplified) one-hot encoded (min 10 occurrences); optional **pattern groups** (reversal/continuation/breakout/compression) and strength (wick ratio, consolidation length).

**XGBoost interaction constraints:** COT features interact only with trend/vol; pattern features only with trend/vol (no COT↔pattern).

---

## 4. Models and Training

- **5-day return model:** Single walk-forward (5 folds, expanding window, 50% min train), target = `next_5d_return`. GridSearchCV (n_estimators, learning_rate, max_depth, subsample), MAE. This is the **only** model; used for trade decisions and calibration.
- **Feature pruning:** After 5d WF, mean and CV of feature importance across folds; drop features below 5th percentile of mean importance or with importance CV > 2.0. Last run dropped 13 features.
- **Calibration:** Isotonic regression on OOS (predicted 5d return → realized 5d return). All **threshold and sizing** use **calibrated** predictions.
- **Threshold/sizing:** Grid search over threshold (0.10%–0.30%) and max confidence cap (2.0–3.0) on **calibrated** OOS; objective = Sharpe − 0.5×max drawdown. Last run: **threshold = 0.30%, max_cap = 2.0**.
- **Optional:** Meta “trade vs no trade” classifier (calibrated pred + vol regime + COT flags); currently off (`USE_TRADE_GATE = False`).

---

## 5. Trade Simulation (backtest)

- **Filter:** Trade only when |calibrated 5d pred| > optimized threshold (0.30%).
- **Sizing:** Base notional × min(confidence, max_cap), confidence = |pred| / median(|pred|).
- **Exit rules:** Target = close reaches predicted level; stop = loss of 1.5×|pred_return| (close-based); else exit at window end.
- **Windows reported:** 5, 10, 15 days (same 5d prediction, different hold periods).

---

## 6. Latest Run Results (representative)

**Data:** COT-only, 5,139 rows → valid rows for 5d model (~4,925). Final holdout: 493 bars (last 20% of OOS); threshold/cap optimized on 1,970 bars only. Execution haircut: 1.0 bps round-trip.

**5-day model:**

- OOS MAE: 0.007509, directional accuracy: 66.1%.
- After isotonic calibration: MAE 0.007347, dir. acc. 67.1%.
- Optimized (on holdout-opt subset): threshold **0.35%**, max_cap **2.0**.

**Simulated performance (5d-model, calibrated, full OOS):**

| Window | Trades | Surpass % | Account % | Sharpe | MaxDD | Daily P&L Sharpe | Daily P&L MaxDD |
|--------|--------|------------|-----------|--------|-------|------------------|-----------------|
| 5d     | 1529   | 71.9%      | +72.35%   | 7.44   | 0.9%  | 3.43             | 2.4%            |
| 10d    | 1529   | 78.0%      | +80.49%   | 7.96   | 0.8%  | 3.28             | 3.9%            |
| 15d    | 1529   | 79.1%      | +82.64%   | 8.11   | 0.8%  | 3.22             | 3.4%            |

- **Directional note:** OOS dir. acc. and 5d surpass % can be ~0.5–1 point lower in some runs see time-slice and holdout tables. Longer windows (10d/15d) often show *higher* surpass in the same run (e.g. 77.3% and 81.6% vs 75.3% and 79.5%). So “slightly lower” on 5d is normal variance; check 10d/15d and Sharpe/MaxDD for consistency.
- LONG: 763 trades (e.g. 71.4% surpass at 5d); SHORT: 766 (72.3% at 5d). Exits: target / stop / expire reported.
- **Daily P&L curve** (mark-to-market by day): Sharpe and MaxDD are lower than per-trade metrics, as they capture overlapping positions and daily equity path.

**Holdout (unseen; threshold/cap not tuned on this period):**

| Window | Trades | Surpass % | Account % | Sharpe | MaxDD | Daily P&L Sharpe | Daily P&L MaxDD |
|--------|--------|------------|-----------|--------|-------|------------------|-----------------|
| 5d     | 298    | 68.1%      | +10.92%   | 6.03   | 0.5%  | 3.20             | 0.8%            |
| 10d    | 298    | 75.8%      | +12.76%   | 6.79   | 0.5%  | 3.48             | 0.9%            |
| 15d    | 298    | 75.8%      | +12.51%   | 6.60   | 0.5%  | 3.33             | 0.9%            |

**A/B threshold (5d window):** 0.25% → 1705 trades, 70.1% surpass, +72.07%, Sharpe 6.91, MaxDD 0.9%; 0.30% → 1576, 71.4%, +72.29%, 7.28, 0.9%; 0.35% → 1529, 71.9%, +72.35%, 7.44, 0.9%.

**Time-slice (5d window):** early 553 trades, 71.1% surpass, +24.80%, Sharpe 6.71, MaxDD 0.9%; mid 470, 73.6%, +27.56%, 8.99, 0.9%; late 506, 71.1%, +22.04%, 7.03, 1.1%.

**Calibration by vol regime (tertiles of atr_pct):** low_vol mean pred +0.056% vs real +0.078% (bias −0.022); mid +0.034% vs +0.046% (−0.013); high −0.022% vs −0.057% (+0.035).

**Production-style output (example):**

- Current close: 1.91252. Raw 5d predicted return: 0.1440%; calibrated: 0.2187% → 1.91670.
- **Signal:** NO TRADE (|0.2187%| &lt; 0.35% threshold).

---

---


## 9. Tech Stack (for reference)

- **Language:** Python 3.
- **Libraries:** pandas, numpy, scikit-learn (IsotonicRegression, LogisticRegression, GridSearchCV, TimeSeriesSplit), XGBoost, scipy (mannwhitneyu).
- **Script:** `holdout_analysis.py` (single main script: load → merge COT → features → WF → 5d model → calibrate → optimize threshold/sizing → trade sim → production-style print).
- **Related:** `timeframe_holdout_analysis.py` does weekly/monthly bars with the same feature and constraint design.

---

#