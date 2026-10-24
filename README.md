# FX Trader

Daily FX prediction and backtest pipeline for five pairs: **GBPAUD, GBPUSD, USDCHF, USDJPY, NZDJPY**.

It loads combined OHLC + pattern + indicator files, merges CFTC TFF (Commitments of Traders) features, trains an XGBoost 5-day return model with walk-forward validation, then runs a trade simulation.

Repo: https://github.com/alagarde-bit/Trading

## What you need

- Python 3.10+ (3.11 used in development)
- Optional: [Rust](https://rustup.rs/) only if you want to **rebuild** chart/candlestick pattern CSVs
- Optional: an [Alpha Vantage](https://www.alphavantage.co/support/#api-key) API key only if you want to **refresh** raw FX CSVs

You do **not** need an API key or Rust to run the model. Combined pair files and a TFF COT file are already in the repo.

## Setup

```bash
git clone https://github.com/alagarde-bit/Trading.git
cd Trading
```

If this project lives in a subfolder on your machine, `cd` into it (for example `FX_Trader`).

```bash
python -m venv .venv
```

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

LSTM scripts (`nzdjpy_lstm_predictor.py` and related) need extra packages:

```bash
pip install -r requirements-lstm.txt
```

## Run the daily model

From the directory that contains `holdout_analysis.py` and `combined_*_daily.csv`:

```bash
python holdout_analysis.py
```

Expected inputs (already committed):

- `combined_{PAIR}_daily.csv` for each pair
- `TFF_-_Futures_Only_20260227.csv` (update `TFF_PATH` in `holdout_analysis.py` if you use a newer TFF dump)

Weekly/monthly analysis:

```bash
python timeframe_holdout_analysis.py
```

Walk-forward GridSearch on ~22k rows is slow (often tens of minutes). That is expected.

## Optional: refresh FX prices

Set `ALPHA_VANTAGE_API_KEY` (see `.env.example`). Alpha Vantage rate-limits; the fetch scripts sleep between pairs.

Windows (PowerShell):

```powershell
$env:ALPHA_VANTAGE_API_KEY = "your_key_here"
python daily_fx.py
python weekly_fx.py
python monthly_fx.py
```

macOS / Linux:

```bash
export ALPHA_VANTAGE_API_KEY=your_key_here
python daily_fx.py
python weekly_fx.py
python monthly_fx.py
```

## Optional: rebuild pattern files (Rust)

Run from `chart_pattern_detection/` so paths like `../GBPAUD_daily.csv` resolve to the project root.

```bash
cd chart_pattern_detection
cargo build --release
```

Then run binaries in this order (each processes all five pairs):

```bash
cargo run --release --bin chart_pattern_historical_metrics
cargo run --release --bin chart_patterns_basic
cargo run --release --bin chart_patterns_basic2
cargo run --release --bin chart_patterns_basic3
cargo run --release --bin chart_patterns_basic4
cargo run --release --bin chart_pattern_detection_hs
cargo run --release --bin candlestick_detection
cargo run --release --bin candlestick_double_detection
cargo run --release --bin candlestick_triple_detection
cargo run --release --bin merge_patterns_script
cargo run --release --bin merge_candlesticks
cargo run --release --bin add_historical_patterns
cargo run --release --bin add_historical_candlesticks
cargo run --release --bin combine_patterns_and_indicators
```

That writes `combined_{PAIR}_daily.csv` in the project root.

## Project layout

| Path | Role |
|------|------|
| `holdout_analysis.py` | Daily 5-day model, calibration, trade sim |
| `timeframe_holdout_analysis.py` | Weekly/monthly models |
| `daily_fx.py` / `weekly_fx.py` / `monthly_fx.py` | Alpha Vantage OHLC fetch |
| `chart_pattern_detection/` | Rust pattern + indicator combine pipeline |
| `combined_*_daily.csv` | Model inputs (OHLC, patterns, SMAs, RSI) |
| `TFF_-_Futures_Only_*.csv` | CFTC TFF COT |
| `FX_MODEL_OVERVIEW.md` | Model/backtest notes |

## Notes for collaborators

- Do not commit `.env` or API keys.
- If you previously cloned an older copy of this repo, pull `master` before running.
- `n_jobs=-1` in GridSearch can use all CPU cores.
