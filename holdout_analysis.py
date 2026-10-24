import warnings
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# --- TFF (CFTC Commitments of Traders) integration ---
# Use latest downloaded TFF file
TFF_PATH = Path('TFF_-_Futures_Only_20260227.csv')
TRAIN_COT_ONLY = True
COT_START_DATE = None
TFF_NUMERIC_COLS = [
    'Open_Interest_All',
    'Pct_of_OI_Dealer_Long_All', 'Pct_of_OI_Dealer_Short_All',
    'Pct_of_OI_Asset_Mgr_Long_All', 'Pct_of_OI_Asset_Mgr_Short_All',
    'Pct_of_OI_Lev_Money_Long_All', 'Pct_of_OI_Lev_Money_Short_All',
]
COT_ZSCORE_WINDOW = 26

# --- Model configuration ---
MIN_PATTERN_OCC = 10
USE_PATTERN_GROUPS_ONLY = False  # True = use only pattern group counts + strength scores; False = keep one-hots too
USE_INTERACTION_CONSTRAINTS = True  # COT<->trend/vol, pattern<->range/body/vol only
DROP_UNSTABLE_FEATURES = True      # Drop low-importance or wildly unstable features across folds
IMPORTANCE_LOW_PCT = 5             # Drop features below this percentile of mean importance
IMPORTANCE_CV_MAX = 2.0             # Drop features with (std/mean) importance > this (unstable)
N_WF_FOLDS = 5
MIN_TRAIN_FRAC = 0.50
# Direction classification: threshold for up/down/flat (e.g. 0.001 ~ 10 pips)
DIRECTION_THRESHOLD = 0.001
USE_DIRECTION_CLASSIFIER = False
# Trade realism
MIN_TRADE_THRESHOLD = 0.002   # only trade when |pred_return| > this (~20 pips GBPAUD)
STOP_LOSS_MULT = 1.5          # stop-loss = STOP_LOSS_MULT * |pred_return|
BASE_TRADE_SIZE = 10000       # base notional; scaled by confidence
MAX_CONFIDENCE_CAP = 3.0      # max multiplier for size (confidence = |pred|/median(|pred|))

# Calibration and decision layer
USE_CALIBRATION = True        # fit isotonic regression: raw pred -> realized avg; use for threshold/sizing
USE_TRADE_GATE = False        # optional meta "trade vs no trade" classifier as additional gate
TRADE_WINDOW_DAYS = 5         # primary window for threshold/sizing optimization (5d model)
# Robustness (Perplexity-style): A/B thresholds, execution haircut, time-slice, calibration diagnostics
RUN_AB_THRESHOLD_SCENARIOS = True   # report 5d sim at 0.25%, 0.30%, 0.35% for comparison
EXECUTION_HAIRCUT_PCT = 0.0001      # round-trip cost as fraction of notional (0.0001 = 1 bp); set 0 to disable
RUN_TIME_SLICE_ANALYSIS = True      # report performance on early/mid/late third of OOS
RUN_CALIBRATION_DIAGNOSTICS = True  # bin by vol regime, report mean pred vs realized per bin
# Final holdout: optimize threshold/cap on first (1-HOLDOUT_FRAC) of OOS; report metrics on unseen holdout
USE_FINAL_HOLDOUT = True
HOLDOUT_FRAC = 0.2                  # last 20% of OOS by time = holdout (unseen)

PAIRS = ['GBPAUD', 'NZDJPY', 'GBPUSD', 'USDCHF', 'USDJPY']

# Fast iteration: FX_TRADER_FAST=1 reduces walk-forward work (for development, not final results)
if os.environ.get("FX_TRADER_FAST", "").strip().lower() in ("1", "true", "yes"):
    N_WF_FOLDS = 2
    MIN_TRAIN_FRAC = 0.80
    RUN_AB_THRESHOLD_SCENARIOS = False
    RUN_TIME_SLICE_ANALYSIS = False
    RUN_CALIBRATION_DIAGNOSTICS = False
    print("FX_TRADER_FAST=1: using 2 walk-forward folds and skipping extra diagnostics.")


def _parse_tff_number(s):
    if pd.isna(s) or s == '' or str(s).strip() == '':
        return np.nan
    return float(str(s).replace(',', '').strip())

def _derive_cot_features(sub, label):
    derived = pd.DataFrame({'report_date': sub['report_date']})
    groups = ['Dealer', 'Asset_Mgr', 'Lev_Money']
    for g in groups:
        long_col = f'cot_{label}_Pct_of_OI_{g}_Long_All'
        short_col = f'cot_{label}_Pct_of_OI_{g}_Short_All'
        if long_col not in sub.columns or short_col not in sub.columns:
            continue
        net_col = f'cot_{label}_{g}_net'
        derived[net_col] = sub[long_col].values - sub[short_col].values
        chg_col = f'cot_{label}_{g}_net_chg'
        derived[chg_col] = derived[net_col].diff()
        z_col = f'cot_{label}_{g}_net_z'
        roll_mean = derived[net_col].rolling(COT_ZSCORE_WINDOW, min_periods=4).mean()
        roll_std = derived[net_col].rolling(COT_ZSCORE_WINDOW, min_periods=4).std()
        derived[z_col] = (derived[net_col] - roll_mean) / roll_std.replace(0, np.nan)

    oi_col = f'cot_{label}_Open_Interest_All'
    if oi_col in sub.columns:
        derived[f'cot_{label}_oi_chg_pct'] = sub[oi_col].values / sub[oi_col].shift(1).values - 1

    return derived

# TFF contract name (CONTRACT_MARKET_NAME) per currency label; USD has no separate contract in this TFF
TFF_CONTRACT_MAP = {
    'gbp': 'BRITISH POUND',
    'aud': 'AUSTRALIAN DOLLAR',
    'chf': 'SWISS FRANC',
    'jpy': 'JAPANESE YEN',
    'nzd': 'NZ DOLLAR',
}
# For each FX pair, which two COT currencies apply (order: base, quote). USD side has no contract → use None
PAIR_TO_CURRENCIES = {
    'GBPAUD': ('gbp', 'aud'),
    'GBPUSD': ('gbp', None),
    'USDCHF': (None, 'chf'),
    'USDJPY': (None, 'jpy'),
    'NZDJPY': ('nzd', 'jpy'),
}


def load_and_merge_tff(daily_df, tff_path=TFF_PATH, commodity_filter=None):
    if not Path(tff_path).exists():
        return daily_df, []

    out_df = daily_df.copy()
    if 'pair' not in out_df.columns:
        out_df['pair'] = 'GBPAUD'

    tff = pd.read_csv(tff_path)
    name_col = 'CONTRACT_MARKET_NAME' if 'CONTRACT_MARKET_NAME' in tff.columns else 'Market_and_Exchange_Names'
    if name_col == 'Market_and_Exchange_Names':
        tff['_contract'] = tff[name_col].str.extract(r'^([A-Z\s]+)\s*-', expand=False).str.strip()
    else:
        tff['_contract'] = tff[name_col].astype(str).str.strip()

    date_col = 'Report_Date_as_YYYY_MM_DD'
    tff[date_col] = pd.to_datetime(tff[date_col], format='%Y %b %d %I:%M:%S %p', errors='coerce')
    tff = tff.dropna(subset=[date_col])

    cot_feature_names = []
    derived_by_label = {}

    for label, commodity in TFF_CONTRACT_MAP.items():
        if commodity_filter is not None and commodity not in commodity_filter:
            continue
        sub = tff.loc[tff['_contract'] == commodity].copy()
        if sub.empty:
            continue
        for c in TFF_NUMERIC_COLS:
            if c not in sub.columns:
                continue
            sub[c] = sub[c].apply(_parse_tff_number)
        sub = sub.dropna(subset=[date_col]).sort_values(date_col)
        cols = [date_col] + [c for c in TFF_NUMERIC_COLS if c in sub.columns]
        sub = sub[cols].drop_duplicates(subset=[date_col], keep='last')
        sub = sub.rename(columns={date_col: 'report_date', **{c: f'cot_{label}_{c}' for c in cols if c != date_col}})

        derived = _derive_cot_features(sub, label)
        derived_by_label[label] = derived
        for c in derived.columns:
            if c != 'report_date':
                cot_feature_names.append(c)
        cot_feature_names.append(f'cot_{label}_available')

    if not derived_by_label:
        return out_df, []

    # Build (date, pair) COT table: each row gets COT for that pair's two currencies only
    unique_dates = out_df['date'].drop_duplicates().sort_values().reset_index(drop=True)
    all_pairs = out_df['pair'].drop_duplicates().tolist()
    all_cot_cols = [c for c in cot_feature_names if not c.endswith('_available')]

    date_to_cot = {}
    all_cot_cols_set = set()
    for label, derived in derived_by_label.items():
        feat_cols = [c for c in derived.columns if c != 'report_date']
        all_cot_cols_set.update(feat_cols)
        merged = pd.merge_asof(
            unique_dates.to_frame(name='date'),
            derived[['report_date'] + feat_cols],
            left_on='date',
            right_on='report_date',
            direction='backward'
        )
        merged = merged.drop_duplicates(subset=['date'], keep='last')
        date_to_cot[label] = merged.set_index('date')[feat_cols]

    for label in derived_by_label:
        all_cot_cols_set.add(f'cot_{label}_available')

    rows = []
    for (date, pair) in out_df[['date', 'pair']].drop_duplicates().itertuples(index=False):
        c1, c2 = PAIR_TO_CURRENCIES.get(pair, (None, None))
        row = {'date': date, 'pair': pair}
        for col in all_cot_cols_set:
            row[col] = np.nan
        for label in (c1, c2):
            if label is None or label not in derived_by_label:
                continue
            ts = date_to_cot.get(label)
            if ts is None:
                continue
            try:
                idx = ts.index.asof(pd.Timestamp(date))
                if pd.isna(idx):
                    continue
                for col in ts.columns:
                    row[col] = ts.loc[idx, col]
            except Exception:
                pass
        for label in derived_by_label:
            filled = [k for k in row if isinstance(k, str) and k.startswith(f'cot_{label}_') and not k.endswith('_available') and pd.notna(row.get(k))]
            row[f'cot_{label}_available'] = 1.0 if filled else 0.0
        rows.append(row)

    cot_table = pd.DataFrame(rows)
    drop_cols = [c for c in cot_table.columns if c not in ('date', 'pair') and c in out_df.columns]
    out_df = out_df.drop(columns=drop_cols, errors='ignore')
    out_df = out_df.merge(cot_table, on=['date', 'pair'], how='left')

    # Pair-specific diff and derived COT features (build many columns; avoid fragmentation warning by copying once at end)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=pd.errors.PerformanceWarning)
        _apply_cot_derived_features(out_df, derived_by_label, cot_feature_names)

    out_df = out_df.copy()
    # Deduplicate feature names
    seen = set()
    unique_cot = []
    for c in cot_feature_names:
        if c not in seen:
            unique_cot.append(c)
            seen.add(c)
    cot_feature_names.clear()
    cot_feature_names.extend(unique_cot)

    for c in cot_feature_names:
        if c in out_df.columns and not c.endswith('_available'):
            if out_df[c].isna().any():
                out_df[c] = out_df[c].fillna(0.0)

    return out_df, cot_feature_names


def _apply_cot_derived_features(out_df, derived_by_label, cot_feature_names):
    """Apply pair-specific diff, lev_minus_am, 4w/8w/z, momentum, 52w_pct (mutates out_df)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=pd.errors.PerformanceWarning)
        _apply_cot_derived_features_impl(out_df, derived_by_label, cot_feature_names)


def _apply_cot_derived_features_impl(out_df, derived_by_label, cot_feature_names):
    for g in ['Dealer', 'Asset_Mgr', 'Lev_Money']:
        diff_net = f'cot_diff_{g}_net'
        diff_chg = f'cot_diff_{g}_net_chg'
        out_df[diff_net] = np.nan
        out_df[diff_chg] = np.nan
        for pair, (c1, c2) in PAIR_TO_CURRENCIES.items():
            if not c1 or not c2:
                continue
            n1, n2 = f'cot_{c1}_{g}_net', f'cot_{c2}_{g}_net'
            ch1, ch2 = f'cot_{c1}_{g}_net_chg', f'cot_{c2}_{g}_net_chg'
            if n1 in out_df.columns and n2 in out_df.columns:
                mask = out_df['pair'] == pair
                out_df.loc[mask, diff_net] = out_df.loc[mask, n1] - out_df.loc[mask, n2]
                if diff_net not in cot_feature_names:
                    cot_feature_names.append(diff_net)
            if ch1 in out_df.columns and ch2 in out_df.columns:
                mask = out_df['pair'] == pair
                out_df.loc[mask, diff_chg] = out_df.loc[mask, ch1] - out_df.loc[mask, ch2]
                if diff_chg not in cot_feature_names:
                    cot_feature_names.append(diff_chg)

    # Lev minus AM per currency, then pair-specific cot_diff_lev_minus_am
    for label in derived_by_label:
        lev_col = f'cot_{label}_Lev_Money_net'
        am_col = f'cot_{label}_Asset_Mgr_net'
        if lev_col in out_df.columns and am_col in out_df.columns:
            out_df[f'cot_{label}_lev_minus_am'] = out_df[lev_col] - out_df[am_col]
            cot_feature_names.append(f'cot_{label}_lev_minus_am')

    out_df['cot_diff_lev_minus_am'] = np.nan
    for pair, (c1, c2) in PAIR_TO_CURRENCIES.items():
        if not c1 or not c2:
            continue
        l1, l2 = f'cot_{c1}_lev_minus_am', f'cot_{c2}_lev_minus_am'
        if l1 in out_df.columns and l2 in out_df.columns:
            mask = out_df['pair'] == pair
            out_df.loc[mask, 'cot_diff_lev_minus_am'] = out_df.loc[mask, l1] - out_df.loc[mask, l2]
    cot_feature_names.append('cot_diff_lev_minus_am')

    # 4w/8w chg and z per currency (within each pair's time series)
    for label in derived_by_label:
        for col in [f'cot_{label}_lev_minus_am', f'cot_{label}_Dealer_net', f'cot_{label}_Asset_Mgr_net', f'cot_{label}_Lev_Money_net']:
            if col not in out_df.columns:
                continue
            for suff, period in [('_4w_chg', 20), ('_8w_chg', 40)]:
                out_df[col + suff] = out_df.groupby('pair')[col].transform(lambda s: s - s.shift(period))
                cot_feature_names.append(col + suff)
            roll_mean = out_df.groupby('pair')[col].transform(lambda s: s.rolling(COT_ZSCORE_WINDOW, min_periods=4).mean())
            roll_std = out_df.groupby('pair')[col].transform(lambda s: s.rolling(COT_ZSCORE_WINDOW, min_periods=4).std())
            out_df[col + '_z'] = (out_df[col] - roll_mean) / roll_std.replace(0, np.nan)
            cot_feature_names.append(col + '_z')
    for col in ['cot_gbp_lev_minus_am', 'cot_aud_lev_minus_am', 'cot_diff_lev_minus_am']:
        if col not in out_df.columns:
            continue
        for suff, period in [('_4w_chg', 20), ('_8w_chg', 40)]:
            if col + suff not in out_df.columns:
                out_df[col + suff] = out_df.groupby('pair')[col].transform(lambda s: s - s.shift(period))
                cot_feature_names.append(col + suff)
        roll_mean = out_df.groupby('pair')[col].transform(lambda s: s.rolling(COT_ZSCORE_WINDOW, min_periods=4).mean())
        roll_std = out_df.groupby('pair')[col].transform(lambda s: s.rolling(COT_ZSCORE_WINDOW, min_periods=4).std())
        out_df[col + '_z'] = (out_df[col] - roll_mean) / roll_std.replace(0, np.nan)
        cot_feature_names.append(col + '_z')

    # COT 4w/8w momentum and 52w percentile per pair (rolling within pair)
    def _pct_rank_52w(s):
        if s.isna().all() or len(s) < 52:
            return np.nan
        return (s.iloc[-1] >= s).mean() * 100

    for label in derived_by_label:
        for g in ['Dealer', 'Asset_Mgr', 'Lev_Money']:
            col = f'cot_{label}_{g}_net'
            if col not in out_df.columns:
                continue
            out_df[f'{col}_4w_chg'] = out_df.groupby('pair')[col].transform(lambda s: s - s.shift(20))
            out_df[f'{col}_8w_chg'] = out_df.groupby('pair')[col].transform(lambda s: s - s.shift(40))
            out_df[f'{col}_momentum'] = out_df[f'{col}_4w_chg'] - out_df[f'{col}_8w_chg']
            out_df[f'{col}_52w_pct'] = out_df.groupby('pair')[col].transform(lambda s: s.rolling(260, min_periods=52).apply(_pct_rank_52w, raw=False))
            cot_feature_names.extend([f'{col}_4w_chg', f'{col}_8w_chg', f'{col}_momentum', f'{col}_52w_pct'])


def compute_relative_features(df):
    """Replace raw prices with stationary relative features."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=pd.errors.PerformanceWarning)
        _compute_relative_features_impl(df)
    return df.copy()

def _compute_relative_features_impl(df):
    df['daily_return'] = df['daily_close'].pct_change()
    df['daily_range_pct'] = (df['daily_high'] - df['daily_low']) / df['daily_close']
    df['daily_body_pct'] = (df['daily_close'] - df['daily_open']) / df['daily_close']
    # Pattern strength: wick ratio (wick length / range), consolidation length (count of low-range bars)
    rng = df['daily_high'] - df['daily_low']
    body_abs = (df['daily_close'] - df['daily_open']).abs()
    df['daily_wick_ratio'] = np.where(rng > 0, (rng - body_abs) / rng, 0)
    med_range_20 = df['daily_range_pct'].rolling(20).median()
    df['consolidation_len_20'] = (df['daily_range_pct'] < med_range_20).astype(float).rolling(20).sum()
    df['close_vs_daily_sma20'] = df['daily_close'] / df['daily_sma_20'] - 1
    df['close_vs_daily_sma50'] = df['daily_close'] / df['daily_sma_50'] - 1
    df['close_vs_daily_sma200'] = df['daily_close'] / df['daily_sma_200'] - 1
    df['daily_sma20_vs_sma50'] = df['daily_sma_20'] / df['daily_sma_50'] - 1
    df['daily_sma50_vs_sma200'] = df['daily_sma_50'] / df['daily_sma_200'] - 1

    df['weekly_range_pct'] = (df['weekly_high'] - df['weekly_low']) / df['weekly_close']
    df['weekly_body_pct'] = (df['weekly_close'] - df['weekly_open']) / df['weekly_close']
    df['close_vs_weekly_sma4'] = df['weekly_close'] / df['weekly_sma_4'] - 1
    df['close_vs_weekly_sma10'] = df['weekly_close'] / df['weekly_sma_10'] - 1
    df['close_vs_weekly_sma40'] = df['weekly_close'] / df['weekly_sma_40'] - 1
    df['weekly_sma4_vs_sma10'] = df['weekly_sma_4'] / df['weekly_sma_10'] - 1
    df['weekly_sma10_vs_sma40'] = df['weekly_sma_10'] / df['weekly_sma_40'] - 1

    df['monthly_range_pct'] = (df['monthly_high'] - df['monthly_low']) / df['monthly_close']
    df['monthly_body_pct'] = (df['monthly_close'] - df['monthly_open']) / df['monthly_close']
    df['close_vs_monthly_sma1'] = df['monthly_close'] / df['monthly_sma_1'] - 1
    df['close_vs_monthly_sma3'] = df['monthly_close'] / df['monthly_sma_3'] - 1
    df['close_vs_monthly_sma10'] = df['monthly_close'] / df['monthly_sma_10'] - 1
    df['monthly_sma1_vs_sma3'] = df['monthly_sma_1'] / df['monthly_sma_3'] - 1
    df['monthly_sma3_vs_sma10'] = df['monthly_sma_3'] / df['monthly_sma_10'] - 1

    # Volatility (ATR via pandas; no talib)
    prev_close = df['daily_close'].shift(1)
    tr = np.maximum(
        df['daily_high'] - df['daily_low'],
        np.maximum(
            (df['daily_high'] - prev_close).abs(),
            (df['daily_low'] - prev_close).abs()
        )
    )
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr_14'] / df['daily_close']
    daily_std_20 = df['daily_close'].rolling(20).std()
    df['bb_width'] = (df['daily_sma_20'] + 2 * daily_std_20 - (df['daily_sma_20'] - 2 * daily_std_20)) / df['daily_sma_20']
    df['realized_vol_5'] = df['daily_return'].rolling(5).std() * np.sqrt(252)
    df['realized_vol_20'] = df['daily_return'].rolling(20).std() * np.sqrt(252)
    df['vol_ratio'] = df['realized_vol_5'] / df['realized_vol_20'].replace(0, np.nan)

    # Momentum
    df['roc_5'] = df['daily_close'].pct_change(5)
    df['roc_20'] = df['daily_close'].pct_change(20)
    df['macd'] = df['daily_close'].ewm(span=12, adjust=False).mean() - df['daily_close'].ewm(span=26, adjust=False).mean()
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # --- First differences (speed) ---
    df['daily_rsi_chg'] = df['daily_rsi'].diff()
    df['weekly_rsi_chg'] = df['weekly_rsi'].diff()
    df['monthly_rsi_chg'] = df['monthly_rsi'].diff()
    df['atr_pct_chg'] = df['atr_pct'].diff()
    df['bb_width_chg'] = df['bb_width'].diff()
    df['realized_vol_5_chg'] = df['realized_vol_5'].diff()
    df['realized_vol_20_chg'] = df['realized_vol_20'].diff()
    df['close_vs_sma20_chg'] = df['close_vs_daily_sma20'].diff()
    df['close_vs_sma50_chg'] = df['close_vs_daily_sma50'].diff()
    df['close_vs_sma200_chg'] = df['close_vs_daily_sma200'].diff()
    for col in ['cot_gbp_Dealer_net_z', 'cot_gbp_Asset_Mgr_net_z', 'cot_gbp_Lev_Money_net_z',
                'cot_aud_Dealer_net_z', 'cot_aud_Asset_Mgr_net_z', 'cot_aud_Lev_Money_net_z']:
        if col in df.columns:
            df[f'{col}_chg'] = df[col].diff()
    for col in ['cot_diff_Dealer_net', 'cot_diff_Asset_Mgr_net', 'cot_diff_Lev_Money_net']:
        if col in df.columns:
            df[f'{col}_chg'] = df[col].diff()

    # --- Second differences (acceleration) ---
    df['daily_rsi_acc'] = df['daily_rsi_chg'].diff()
    df['atr_pct_acc'] = df['atr_pct_chg'].diff()
    for col in ['cot_gbp_Dealer_net_z', 'cot_gbp_Asset_Mgr_net_z', 'cot_aud_Dealer_net_z']:
        if f'{col}_chg' in df.columns:
            df[f'{col}_acc'] = df[f'{col}_chg'].diff()

    # --- HAR-style multi-scale (daily / 5d avg / 22d avg) ---
    df['return_5d_avg'] = df['daily_return'].rolling(5).mean()
    df['return_22d_avg'] = df['daily_return'].rolling(22).mean()
    df['realized_vol_22'] = df['daily_return'].rolling(22).std() * np.sqrt(252)
    df['rv_5d_avg'] = df['realized_vol_5'].rolling(5).mean()
    df['rv_22d_avg'] = df['realized_vol_20'].rolling(22).mean()
    df['atr_pct_5d_avg'] = df['atr_pct'].rolling(5).mean()
    df['atr_pct_22d_avg'] = df['atr_pct'].rolling(22).mean()

    # --- Area / persistence (rolling sum of close vs SMA) ---
    df['area_sma20_10d'] = df['close_vs_daily_sma20'].rolling(10).sum()
    df['area_sma20_20d'] = df['close_vs_daily_sma20'].rolling(20).sum()
    df['area_sma50_10d'] = df['close_vs_daily_sma50'].rolling(10).sum()
    df['area_sma50_20d'] = df['close_vs_daily_sma50'].rolling(20).sum()
    df['area_sma200_10d'] = df['close_vs_daily_sma200'].rolling(10).sum()
    df['area_sma200_20d'] = df['close_vs_daily_sma200'].rolling(20).sum()
    # COT z-score area and extreme count (4w=20d, 8w=40d, 12w=60d)
    for col in ['cot_gbp_Dealer_net_z', 'cot_aud_Dealer_net_z', 'cot_gbp_Lev_Money_net_z']:
        if col not in df.columns:
            continue
        for w, days in [('4w', 20), ('8w', 40), ('12w', 60)]:
            df[f'{col}_sum_{w}'] = df[col].rolling(days).sum()
            df[f'{col}_extreme_{w}'] = (df[col].abs() > 2).astype(float).rolling(days).sum()

    # --- Autocorrelation and distribution shape ---
    def _roll_autocorr(s, lag=1):
        if s.isna().all() or len(s) < 3:
            return np.nan
        return s.autocorr(lag=lag)
    df['return_autocorr_10d'] = df['daily_return'].rolling(10).apply(_roll_autocorr, raw=False)
    df['return_autocorr_20d'] = df['daily_return'].rolling(20).apply(_roll_autocorr, raw=False)
    df['return_sq_autocorr_10d'] = df['daily_return'].pow(2).rolling(10).apply(_roll_autocorr, raw=False)
    df['return_sq_autocorr_20d'] = df['daily_return'].pow(2).rolling(20).apply(_roll_autocorr, raw=False)
    df['return_skew_10d'] = df['daily_return'].rolling(10).skew()
    df['return_skew_20d'] = df['daily_return'].rolling(20).skew()
    df['return_kurt_10d'] = df['daily_return'].rolling(10).kurt()
    df['return_kurt_20d'] = df['daily_return'].rolling(20).kurt()

    # --- Position pain: COT z-score × recent return (crowded-and-wrong setups) ---
    for zcol in ['cot_gbp_Dealer_net_z', 'cot_gbp_Asset_Mgr_net_z', 'cot_gbp_Lev_Money_net_z',
                 'cot_aud_Dealer_net_z', 'cot_aud_Lev_Money_net_z']:
        if zcol not in df.columns:
            continue
        df[f'{zcol}_pain_5d'] = df[zcol] * df['roc_5']
        df[f'{zcol}_pain_20d'] = df[zcol] * df['roc_20']

    # Lagged returns (memory of recent price action; daily_return is lag 1)
    for lag in [2, 3, 5, 10, 20, 60]:
        df[f'return_lag_{lag}'] = df['daily_close'].pct_change(lag)

    df['daily_vs_weekly_close'] = df['daily_close'] / df['weekly_close'] - 1
    df['weekly_vs_monthly_close'] = df['weekly_close'] / df['monthly_close'] - 1
    df['daily_vs_monthly_close'] = df['daily_close'] / df['monthly_close'] - 1


PATTERN_GROUP_MAP = {
    'reversal': [
        'doji', 'hammer', 'hanging_man', 'shooting_star', 'inverted_hammer', 'gravestone', 'dragonfly',
        'evening_star', 'morning_star', 'headandshoulders', 'inverseheadandshoulders', 'doubletop', 'doublebottom',
        'dark_cloud', 'piercing', 'tweezer'
    ],
    'continuation': [
        'three_white', 'three_black', 'three_inside', 'three_outside', 'flag', 'rising_wedge', 'falling_wedge',
        'risingbottoms', 'fallingtops', 'risingtops', 'fallingbottoms'
    ],
    'breakout': ['breakout', 'channel', 'cupandhandle', 'deadcatbounce', 'tower'],
    'compression': [
        'ascending', 'descending', 'triangle', 'symmetrical', 'rectangle', 'broadening', 'cup_', 'cupandhandle',
        'narrow', 'consolidation'
    ]
}

pattern_grp_cols = [f'pattern_grp_{g}' for g in PATTERN_GROUP_MAP.keys()]

def simplify_pattern(pattern):
    parts = pattern.split('_')
    if len(parts) >= 2:
        return parts[0] + '_' + parts[-1]
    return pattern


def simplify_chart_patterns(cell):
    if pd.isna(cell) or cell == '':
        return ''
    patterns = cell.split(';')
    simplified = [simplify_pattern(p) for p in patterns]
    return ';'.join(simplified)


def _pattern_to_group(name):
    if not name or not isinstance(name, str):
        return None
    n = name.lower().strip()
    for group, keywords in PATTERN_GROUP_MAP.items():
        if any(kw in n for kw in keywords):
            return group
    return None


def add_pattern_group_counts(df, pattern_cols, group_map=PATTERN_GROUP_MAP):
    """Add columns pattern_grp_reversal, pattern_grp_continuation, pattern_grp_breakout, pattern_grp_compression."""
    groups = list(group_map.keys())

    def count_groups(row):
        cnt = {g: 0 for g in groups}
        for col in pattern_cols:
            if col not in row.index:
                continue
            for part in str(row[col]).split(';'):
                grp = _pattern_to_group(part)
                if grp:
                    cnt[grp] += 1
        return pd.Series({f'pattern_grp_{g}': cnt[g] for g in groups})

    df[pattern_grp_cols] = df.apply(count_groups, axis=1)
    return df


def build_interaction_constraints(feature_names, encoded_cols=None):
    """COT <-> trend/vol only; pattern <-> range/body/vol only. Returns list of lists of feature *names* (XGBoost expects names when X is DataFrame)."""
    encoded_cols = encoded_cols or []
    cot_names = [n for n in feature_names if n.startswith('cot_') and not n.endswith('_available')]
    pattern_names = [n for n in feature_names if n.startswith('pattern_grp_') or (n in encoded_cols)]
    trend_vol_names = [n for n in feature_names if n not in cot_names and n not in pattern_names]
    if not cot_names or not trend_vol_names:
        return None
    return [cot_names + trend_vol_names, trend_vol_names + pattern_names]


def check_surpass_and_pnl(row, df_ref, pred_return, days=5, spread=0.0002,
                          trade_size=10000, stop_loss_mult=STOP_LOSS_MULT):
    """Exit with stop-loss; when both stop and target are hit same day (high/low), treat as stop (conservative)."""
    entry_price = row['daily_close']
    pred_close = entry_price * (1 + pred_return)
    stop_dist = abs(pred_return) * stop_loss_mult
    date_idx = row.name
    # Filter to same pair to avoid cross-pair price contamination
    if 'pair' in df_ref.columns and 'pair' in row.index:
        pair_mask = df_ref['pair'] == row['pair']
        df_pair = df_ref.loc[pair_mask]
    else:
        df_pair = df_ref
    future_idx = df_pair.index[df_pair.index > date_idx][:days]
    if len(future_idx) == 0:
        return False, 0.0, 'no_data', date_idx

    is_long = pred_return > 0
    has_hl = 'daily_high' in df_pair.columns and 'daily_low' in df_pair.columns

    for i, idx in enumerate(future_idx):
        close = df_pair.loc[idx, 'daily_close']
        if has_hl:
            high = df_pair.loc[idx, 'daily_high']
            low = df_pair.loc[idx, 'daily_low']
            if is_long:
                stop_level = entry_price * (1 - stop_dist)
                stop_hit = low <= stop_level
                target_hit = high >= pred_close
            else:
                stop_level = entry_price * (1 + stop_dist)
                stop_hit = high >= stop_level
                target_hit = low <= pred_close
            if stop_hit and target_hit:
                pnl = (-stop_dist - spread / entry_price) * trade_size
                return False, pnl, 'stop', idx
            if stop_hit:
                pnl = (-stop_dist - spread / entry_price) * trade_size
                return False, pnl, 'stop', idx
            if target_hit:
                ret = (close - entry_price) / entry_price if is_long else (entry_price - close) / entry_price
                pnl = (ret - spread / entry_price) * trade_size
                return True, pnl, 'target', idx
        else:
            ret = (close - entry_price) / entry_price
            trade_ret = ret if is_long else -ret
            if trade_ret <= -stop_dist:
                pnl = (-stop_dist - spread / entry_price) * trade_size
                return False, pnl, 'stop', idx
            if is_long and close >= pred_close:
                pnl = (ret - spread / entry_price) * trade_size
                return True, pnl, 'target', idx
            if not is_long and close <= pred_close:
                pnl = (-ret - spread / entry_price) * trade_size
                return True, pnl, 'target', idx

    exit_idx = future_idx[-1]
    final_close = df_pair.loc[exit_idx, 'daily_close']
    ret = (final_close - entry_price) / entry_price
    pnl = (ret if is_long else -ret) * trade_size - (spread / entry_price) * trade_size
    surpass = (final_close >= pred_close) if is_long else (final_close <= pred_close)
    return surpass, pnl, 'expire', exit_idx


def _max_drawdown(profits, initial_balance=100000):
    """Compute max drawdown (fraction of peak balance)."""
    balance = initial_balance
    peak = balance
    max_dd = 0.0
    for p in profits:
        balance += p
        peak = max(peak, balance)
        if peak > 0:
            dd = (peak - balance) / peak
            max_dd = max(max_dd, dd)
    return max_dd


def run_trade_sim(all_oos_indices, all_oos_preds, df_oos, days, label="",
                  min_threshold=None, base_size=None, max_conf_cap=None,
                  trade_gate_proba=None, return_metrics=False,
                  execution_haircut_pct=None):
    """Run trade sim with threshold filter, stop-loss, confidence sizing.
    If trade_gate_proba is provided (same length as all_oos_indices), only trade when trade_gate_proba[i] >= 0.5.
    execution_haircut_pct: round-trip cost as fraction of notional (e.g. 0.0001 = 1 bp); applied to each trade.
    If return_metrics=True, returns dict with sharpe, max_drawdown, account_pct, n_trades (no print)."""
    min_threshold = min_threshold if min_threshold is not None else MIN_TRADE_THRESHOLD
    base_size = base_size if base_size is not None else BASE_TRADE_SIZE
    max_conf_cap = max_conf_cap if max_conf_cap is not None else MAX_CONFIDENCE_CAP
    execution_haircut_pct = execution_haircut_pct if execution_haircut_pct is not None else EXECUTION_HAIRCUT_PCT

    initial_balance = 100000
    balance = initial_balance
    trades_taken = 0
    results = {'long': {'surpass': 0, 'count': 0, 'pnl': 0.0, 'profits': []},
               'short': {'surpass': 0, 'count': 0, 'pnl': 0.0, 'profits': []}}
    exit_reasons = {'target': 0, 'stop': 0, 'expire': 0, 'no_data': 0}
    all_profits = []
    trades_for_daily = []  # (entry_idx, exit_idx, is_long, entry_price, trade_size, realized_pnl)

    median_pred = np.median(np.abs(all_oos_preds))

    for i, idx in enumerate(all_oos_indices):
        pred_ret = all_oos_preds[i]
        if abs(pred_ret) < min_threshold:
            continue
        if trade_gate_proba is not None and trade_gate_proba[i] < 0.5:
            continue

        row = df_oos.loc[idx]
        confidence = abs(pred_ret) / median_pred if median_pred > 0 else 1.0
        trade_size = base_size * min(confidence, max_conf_cap)

        surpass, profit, reason, exit_idx = check_surpass_and_pnl(
            row, df_oos, pred_ret, days=days, trade_size=trade_size)
        if execution_haircut_pct > 0:
            profit -= trade_size * execution_haircut_pct

        direction = 'long' if pred_ret > 0 else 'short'
        results[direction]['count'] += 1
        results[direction]['surpass'] += int(surpass)
        results[direction]['pnl'] += profit
        results[direction]['profits'].append(profit)
        all_profits.append(profit)
        balance += profit
        trades_taken += 1
        exit_reasons[reason] += 1
        trades_for_daily.append({
            'entry_idx': idx, 'exit_idx': exit_idx, 'is_long': pred_ret > 0,
            'entry_price': row['daily_close'], 'trade_size': trade_size, 'realized_pnl': profit,
            'pair': row['pair'] if 'pair' in row.index else None
        })

    total_surpass = results['long']['surpass'] + results['short']['surpass']
    surpass_pct = (total_surpass / trades_taken * 100) if trades_taken > 0 else 0
    acct_inc = ((balance - initial_balance) / initial_balance) * 100
    sharpe = (np.mean(all_profits) / np.std(all_profits) * np.sqrt(252)
              if all_profits and np.std(all_profits) > 0 else 0)
    max_dd = _max_drawdown(all_profits, initial_balance)

    # Daily P&L curve: mark open trades to each day's close; add realized PnL on exit day
    sharpe_daily = None
    max_dd_daily = None
    if trades_for_daily and len(df_oos.index) > 0:
        min_entry = min(t['entry_idx'] for t in trades_for_daily)
        max_exit = max(t['exit_idx'] for t in trades_for_daily)
        # Build per-pair index lookup for daily close (avoids cross-pair price contamination)
        _has_pair = 'pair' in df_oos.columns
        if _has_pair:
            _pair_dfs = {p: g for p, g in df_oos.groupby('pair')}
        all_days = sorted(set(d for d in df_oos.index if min_entry <= d <= max_exit))
        if all_days:
            daily_pnl = []
            for d in all_days:
                pnl_d = 0.0
                for t in trades_for_daily:
                    if t['exit_idx'] == d:
                        pnl_d += t['realized_pnl']
                    elif t['entry_idx'] <= d < t['exit_idx']:
                        try:
                            if _has_pair and t['pair'] is not None:
                                pdf = _pair_dfs.get(t['pair'])
                                if pdf is None or d not in pdf.index:
                                    continue
                                close_d = pdf.loc[d, 'daily_close']
                            else:
                                close_d = df_oos.loc[d, 'daily_close']
                            ret_d = (close_d - t['entry_price']) / t['entry_price'] if t['is_long'] else (t['entry_price'] - close_d) / t['entry_price']
                            pnl_d += ret_d * t['trade_size']
                        except (KeyError, TypeError):
                            pass
                daily_pnl.append(pnl_d)
            daily_pnl = np.array(daily_pnl)
            if len(daily_pnl) >= 2 and np.std(daily_pnl) > 0:
                sharpe_daily = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(252)
            equity = initial_balance + np.cumsum(daily_pnl)
            peak = np.maximum.accumulate(equity)
            with np.errstate(invalid='ignore', divide='ignore'):
                dd_daily = np.where(peak > 0, (peak - equity) / peak, 0)
            max_dd_daily = np.nanmax(dd_daily) if len(dd_daily) > 0 else 0.0

    if return_metrics:
        out = {'sharpe': sharpe, 'max_drawdown': max_dd, 'account_pct': acct_inc,
               'n_trades': trades_taken, 'surpass_pct': surpass_pct, 'all_profits': all_profits}
        if sharpe_daily is not None:
            out['sharpe_daily'] = sharpe_daily
        if max_dd_daily is not None:
            out['max_drawdown_daily'] = max_dd_daily
        return out

    print(f"\n{'='*60}")
    print(f"{label}Window: {days} days | Trades taken: {trades_taken}/{len(all_oos_indices)} "
          f"(filtered {len(all_oos_indices) - trades_taken} below threshold)")
    print(f"  Surpass: {surpass_pct:.1f}% | Account: {acct_inc:+.2f}% | Sharpe: {sharpe:.2f} | MaxDD: {max_dd*100:.1f}%")
    if sharpe_daily is not None and max_dd_daily is not None:
        print(f"  (Daily P&L curve: Sharpe={sharpe_daily:.2f}, MaxDD={max_dd_daily*100:.1f}%)")
    print(f"  Exits: target={exit_reasons['target']}, stop={exit_reasons['stop']}, "
          f"expire={exit_reasons['expire']}")

    for d in ['long', 'short']:
        r = results[d]
        if r['count'] == 0:
            continue
        d_surp = r['surpass'] / r['count'] * 100
        d_sharpe = (np.mean(r['profits']) / np.std(r['profits']) * np.sqrt(252)
                    if r['profits'] and np.std(r['profits']) > 0 else 0)
        print(f"  {d.upper():5s}: {r['count']} trades, surpass {d_surp:.1f}%, "
              f"PnL {r['pnl']:+.2f}, Sharpe {d_sharpe:.2f}")


def get_latest_feature_vector(X_cols, df_source, pattern_cols):
    out = []
    for col in X_cols:
        if col in df_source.columns:
            v = df_source[col].iloc[-1]
            out.append(float(v) if pd.notna(v) else 0.0)
        else:
            # One-hot pattern column
            for pcol in pattern_cols:
                if col.startswith(pcol + '_'):
                    pat = col.replace(pcol + '_', '')
                    pats = str(df_source[pcol].iloc[-1] if pcol in df_source.columns else '').split(';')
                    out.append(1.0 if pat in pats else 0.0)
                    break
            else:
                out.append(0.0)
    return np.array(out, dtype=float).reshape(1, -1)


def main():
    # ============================================================
    # Load and preprocess data
    #   - Train on cumulative combined patterns/indicators for all pairs
    #     using the per-pair Rust outputs: combined_{PAIR}_daily.csv
    # ============================================================
    dfs = []
    for pair in PAIRS:
        path = Path(f'combined_{pair}_daily.csv')
        if not path.exists():
            continue
        tmp = pd.read_csv(path, parse_dates=['date'])
        tmp['pair'] = pair
        dfs.append(tmp)

    if not dfs:
        raise FileNotFoundError("No per-pair combined_*_daily.csv files found for PAIRS.")

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(['pair', 'date']).reset_index(drop=True)

    # Optionally materialize the cumulative file for inspection/reuse
    try:
        df.to_csv('combined_all_pairs_daily.csv', index=False)
    except Exception:
        # Non-fatal if we cannot write; training can still proceed
        pass

    df, cot_features = load_and_merge_tff(df)
    # Compute rolling/shift features per pair to avoid cross-pair contamination
    if 'pair' in df.columns and df['pair'].nunique() > 1:
        df = df.groupby('pair', group_keys=False).apply(compute_relative_features).reset_index(drop=True)
    else:
        df = compute_relative_features(df)

    chart_pattern_cols = [
        'daily_chart_patterns', 'daily_chart_prev_pattern', 'daily_chart_prev_3_patterns', 'daily_chart_prev_10_patterns',
        'weekly_chart_patterns', 'weekly_chart_prev_pattern', 'weekly_chart_prev_3_patterns', 'weekly_chart_prev_10_patterns',
        'monthly_chart_patterns', 'monthly_chart_prev_pattern', 'monthly_chart_prev_3_patterns', 'monthly_chart_prev_10_patterns'
    ]
    for col in chart_pattern_cols:
        df[col] = df[col].apply(simplify_chart_patterns)

    if cot_features and TRAIN_COT_ONLY:
        dropped_avail = 0
        # Pair-aware: keep row only if this pair's two COT currencies have available=1
        def _row_has_cot(row):
            c1, c2 = PAIR_TO_CURRENCIES.get(row['pair'], (None, None))
            a1 = row.get(f'cot_{c1}_available', 0) == 1 if c1 else True
            a2 = row.get(f'cot_{c2}_available', 0) == 1 if c2 else True
            return a1 and a2
        if 'pair' in df.columns:
            before = len(df)
            df = df.loc[df.apply(_row_has_cot, axis=1)].copy()
            dropped_avail = before - len(df)
        if COT_START_DATE is not None:
            start = pd.to_datetime(COT_START_DATE)
            before = len(df)
            df = df.loc[df['date'] >= start].copy()
            dropped_date = before - len(df)
            print(f"COT-only mode: from {COT_START_DATE}, using {len(df)} rows (dropped {dropped_avail} without COT, {dropped_date} before start date)")
        else:
            print(f"COT-only mode: using {len(df)} rows (dropped {dropped_avail} without COT)")

    # ============================================================
    # Target: 5-day forward return (only model used for trading)
    # No look-ahead: target uses shift(-5) (future); features use only past/current (pct_change, rolling, shift(1)).
    # Signal and "entry" are at bar-t close; future bars used only for exit.
    # Computed per pair to avoid cross-pair leakage at pair boundaries.
    # ============================================================
    df['next_5d_return'] = df.groupby('pair')['daily_close'].transform(lambda s: s.shift(-5) / s - 1)

    # Re-sort by date for time-series walk-forward (all pairs interleaved chronologically)
    df = df.sort_values('date').reset_index(drop=True)

    # ============================================================
    # Features (all use current or lagged data only)
    # ============================================================
    num_features = [
        'daily_return', 'daily_range_pct', 'daily_body_pct',
        'close_vs_daily_sma20', 'close_vs_daily_sma50', 'close_vs_daily_sma200',
        'daily_sma20_vs_sma50', 'daily_sma50_vs_sma200',
        'daily_rsi',
        'atr_pct', 'bb_width', 'realized_vol_5', 'realized_vol_20', 'vol_ratio',
        'roc_5', 'roc_20', 'macd', 'macd_signal', 'macd_hist',
        # First differences (speed)
        'daily_rsi_chg', 'weekly_rsi_chg', 'monthly_rsi_chg',
        'atr_pct_chg', 'bb_width_chg', 'realized_vol_5_chg', 'realized_vol_20_chg',
        'close_vs_sma20_chg', 'close_vs_sma50_chg', 'close_vs_sma200_chg',
        # Second differences (acceleration)
        'daily_rsi_acc', 'atr_pct_acc',
        # HAR-style multi-scale
        'return_5d_avg', 'return_22d_avg', 'realized_vol_22',
        'rv_5d_avg', 'rv_22d_avg', 'atr_pct_5d_avg', 'atr_pct_22d_avg',
        # Area / persistence
        'area_sma20_10d', 'area_sma20_20d', 'area_sma50_10d', 'area_sma50_20d',
        'area_sma200_10d', 'area_sma200_20d',
        # Autocorr and distribution shape
        'return_autocorr_10d', 'return_autocorr_20d',
        'return_sq_autocorr_10d', 'return_sq_autocorr_20d',
        'return_skew_10d', 'return_skew_20d', 'return_kurt_10d', 'return_kurt_20d',
        # Pattern strength (wick/body, consolidation)
        'daily_wick_ratio', 'consolidation_len_20',
        'return_lag_2', 'return_lag_3', 'return_lag_5', 'return_lag_10', 'return_lag_20', 'return_lag_60',
        'weekly_range_pct', 'weekly_body_pct',
        'close_vs_weekly_sma4', 'close_vs_weekly_sma10', 'close_vs_weekly_sma40',
        'weekly_sma4_vs_sma10', 'weekly_sma10_vs_sma40',
        'weekly_rsi',
        'monthly_range_pct', 'monthly_body_pct',
        'close_vs_monthly_sma1', 'close_vs_monthly_sma3', 'close_vs_monthly_sma10',
        'monthly_sma1_vs_sma3', 'monthly_sma3_vs_sma10',
        'monthly_rsi',
        'daily_vs_weekly_close', 'weekly_vs_monthly_close', 'daily_vs_monthly_close',
    ]
    # Keep only features that exist; add all COT-related columns (from TFF merge + derived _chg, _acc, _sum_*, _extreme_*)
    num_features = [f for f in num_features if f in df.columns]
    num_features = num_features + [f for f in cot_features if f in df.columns and f not in num_features]
    _cot_extra = [c for c in df.columns if c.startswith('cot_') and not c.endswith('_available') and c not in num_features]
    num_features = num_features + _cot_extra
    # Deduplicate while preserving order
    _seen = set()
    num_features = [f for f in num_features if not (f in _seen or _seen.add(f))]

    pattern_cols = [
        'daily_candlestick_patterns', 'daily_candlestick_prev_pattern', 'daily_candlestick_prev_3_patterns', 'daily_candlestick_prev_10_patterns',
        'daily_chart_patterns', 'daily_chart_prev_pattern', 'daily_chart_prev_3_patterns', 'daily_chart_prev_10_patterns',
        'weekly_candlestick_patterns', 'weekly_candlestick_prev_pattern', 'weekly_candlestick_prev_3_patterns', 'weekly_candlestick_prev_10_patterns',
        'weekly_chart_patterns', 'weekly_chart_prev_pattern', 'weekly_chart_prev_3_patterns', 'weekly_chart_prev_10_patterns',
        'monthly_candlestick_patterns', 'monthly_candlestick_prev_pattern', 'monthly_candlestick_prev_3_patterns', 'monthly_candlestick_prev_10_patterns',
        'monthly_chart_patterns', 'monthly_chart_prev_pattern', 'monthly_chart_prev_3_patterns', 'monthly_chart_prev_10_patterns'
    ]

    # Pattern name -> group; keywords must match actual pattern strings used in the model:
    # Candlestick (as-is): Hammer, Hanging_Man, Shooting_Star, *Doji, Three_White_Soldiers_3, etc.
    # Chart (after simplify_chart_patterns): HeadAndShoulders_NN, DoubleTop_NN, BearishFlag_NN, Ascending_NN, etc.

    pattern_grp_cols = [f'pattern_grp_{g}' for g in PATTERN_GROUP_MAP.keys()]
    add_pattern_group_counts(df, pattern_cols)
    num_features = num_features + [c for c in pattern_grp_cols if c in df.columns]

    # Encode patterns with frequency filtering (optional; skipped if USE_PATTERN_GROUPS_ONLY)
    all_encoded = None
    encoded_cols = []
    if not USE_PATTERN_GROUPS_ONLY:
        encoded_dfs = []
        for col in pattern_cols:
            mlb = MultiLabelBinarizer()
            df[col] = df[col].astype(str)
            patterns = df[col].str.split(';').fillna('')
            encoded = mlb.fit_transform(patterns)
            encoded_df = pd.DataFrame(encoded, columns=[f"{col}_{cls}" for cls in mlb.classes_], index=df.index)
            encoded_dfs.append(encoded_df)
        all_encoded = pd.concat(encoded_dfs, axis=1)
        pattern_counts = all_encoded.sum()
        keep_mask = pattern_counts >= MIN_PATTERN_OCC
        dropped_pat = int((~keep_mask).sum())
        kept_pat = int(keep_mask.sum())
        all_encoded = all_encoded.loc[:, keep_mask]
        encoded_cols = list(all_encoded.columns)
        print(f"Pattern features: {kept_pat} kept, {dropped_pat} dropped (< {MIN_PATTERN_OCC} occurrences)")
    else:
        print("Pattern features: using group counts + strength only (USE_PATTERN_GROUPS_ONLY=True)")

    # Combine features
    if all_encoded is not None:
        X = pd.concat([df[num_features], all_encoded], axis=1)
    else:
        X = df[num_features].copy()

    # Drop any duplicate columns (XGBoost crashes if data[col] returns a DataFrame instead of Series)
    if X.columns.duplicated().any():
        dupes = X.columns[X.columns.duplicated()].unique().tolist()
        print(f"Warning: dropping {len(dupes)} duplicate column names: {dupes[:10]}...")
        X = X.loc[:, ~X.columns.duplicated()]

    valid_rows = X.notna().all(axis=1) & df['next_5d_return'].notna()
    X = X.loc[valid_rows]
    df = df.loc[valid_rows]

    # ============================================================
    # 5-day return walk-forward (expanding window)
    # ============================================================
    y_5d = df['next_5d_return']
    n = len(X)
    min_train = int(n * MIN_TRAIN_FRAC)
    test_chunk = max(1, (n - min_train) // N_WF_FOLDS)

    if os.environ.get("FX_TRADER_FAST", "").strip().lower() in ("1", "true", "yes"):
        param_grid = {
            'n_estimators': [80],
            'learning_rate': [0.1],
            'max_depth': [3],
            'subsample': [0.9]
        }
    else:
        param_grid = {
            'n_estimators': [80, 100, 120],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 4],
            'subsample': [0.9, 1.0]
        }

    oos_5d_indices = []
    oos_5d_preds = []
    oos_5d_actuals = []
    last_5d_params = None
    fold_importances = []

    interaction_constraints = None
    if USE_INTERACTION_CONSTRAINTS:
        interaction_constraints = build_interaction_constraints(list(X.columns), encoded_cols)
        if interaction_constraints is not None:
            print(f"Interaction constraints: COT<->trend/vol, pattern<->trend/vol (no COT<->pattern)")

    print(f"\n--- 5-Day Return Walk-Forward ({N_WF_FOLDS} folds, min train {MIN_TRAIN_FRAC*100:.0f}%) ---")
    print(f"Total rows: {n}, min train: {min_train}, test chunk: ~{test_chunk}")

    for fold in range(N_WF_FOLDS):
        train_end = min_train + fold * test_chunk
        test_end = train_end + test_chunk if fold < N_WF_FOLDS - 1 else n

        X_train_fold = X.iloc[:train_end]
        y_train_fold = y_5d.iloc[:train_end]
        X_test_fold = X.iloc[train_end:test_end]
        y_test_fold = y_5d.iloc[train_end:test_end]

        if len(X_test_fold) == 0:
            continue

        # GridSearchCV runs 36 param combos × 3 CV splits = 108 fits per fold; no output until done
        print(f"  Fold {fold+1}: fitting GridSearchCV ({len(X_train_fold)} train, {len(X_test_fold)} test)...", flush=True)

        model_kw = dict(objective='reg:squarederror')
        if interaction_constraints is not None:
            model_kw['interaction_constraints'] = interaction_constraints
        model = xgb.XGBRegressor(**model_kw)
        tscv_inner = TimeSeriesSplit(n_splits=3)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=tscv_inner,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train_fold, y_train_fold)
        last_5d_params = grid_search.best_params_
        fold_importances.append(grid_search.best_estimator_.feature_importances_)

        preds = grid_search.best_estimator_.predict(X_test_fold)
        fold_mae = mean_absolute_error(y_test_fold, preds)
        fold_dir_acc = np.mean(np.sign(preds) == np.sign(y_test_fold.values)) * 100

        print(f"  Fold {fold+1}: Train={train_end}, Test={len(X_test_fold)}, "
              f"MAE={fold_mae:.6f}, Dir Acc={fold_dir_acc:.1f}%")

        oos_5d_indices.extend(X_test_fold.index.tolist())
        oos_5d_preds.extend(preds.tolist())
        oos_5d_actuals.extend(y_test_fold.values.tolist())

    oos_5d_preds = np.array(oos_5d_preds)
    oos_5d_actuals = np.array(oos_5d_actuals)

    # Feature importance stability across folds (from 5d WF)
    if fold_importances and len(fold_importances) >= 2:
        imp_arr = np.array(fold_importances)
        imp_mean = imp_arr.mean(axis=0)
        imp_std = imp_arr.std(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            imp_cv = np.where(imp_mean > 1e-8, imp_std / imp_mean, np.nan)
        low_thresh = np.nanpercentile(imp_mean, IMPORTANCE_LOW_PCT)
        drop_low = imp_mean < low_thresh
        drop_unstable = np.isfinite(imp_cv) & (imp_cv > IMPORTANCE_CV_MAX)
        drop_mask = drop_low | drop_unstable
        stable_names = [X.columns[i] for i in range(len(drop_mask)) if not drop_mask[i]]
        unstable_or_low = [X.columns[i] for i in range(len(drop_mask)) if drop_mask[i]]
        if unstable_or_low and DROP_UNSTABLE_FEATURES:
            print(f"\nFeature importance: dropping {len(unstable_or_low)} low/unstable features (low pct={IMPORTANCE_LOW_PCT}, cv_max={IMPORTANCE_CV_MAX})")
            X = X[stable_names]
            fold_importances = []
        else:
            print(f"\nFeature importance: {X.shape[1]} features; mean importance range [{imp_mean.min():.4f}, {imp_mean.max():.4f}]")

    if cot_features:
        n_cot = len([f for f in cot_features if f in X.columns])
        avail = [f for f in cot_features if f.endswith('_available')]
        print(f"COT/TFF features in model: {n_cot} (incl. availability: {', '.join(avail)})")

    # Align for 5d model and trade sim
    X_5d = X
    y_5d = df['next_5d_return']
    df_oos_5d = df.loc[[i for i in oos_5d_indices if i in df.index]]

    print(f"\n5-Day Aggregate OOS MAE: {mean_absolute_error(oos_5d_actuals, oos_5d_preds):.6f}")
    print(f"5-Day Aggregate OOS Dir Acc: {np.mean(np.sign(oos_5d_preds) == np.sign(oos_5d_actuals)) * 100:.1f}%")

    # ============================================================
    # Trade performance (realistic: threshold + stop-loss + confidence sizing)
    # ============================================================
    # --- Isotonic calibration: raw 5d preds -> realized 5d returns ---
    calibrator_5d = None
    oos_5d_calibrated = oos_5d_preds.copy()
    if USE_CALIBRATION and len(oos_5d_preds) >= 20:
        calibrator_5d = IsotonicRegression(out_of_bounds='clip')
        calibrator_5d.fit(oos_5d_preds, oos_5d_actuals)
        oos_5d_calibrated = calibrator_5d.predict(oos_5d_preds)
        cal_mae = mean_absolute_error(oos_5d_actuals, oos_5d_calibrated)
        cal_dir = np.mean(np.sign(oos_5d_calibrated) == np.sign(oos_5d_actuals)) * 100
        print(f"Calibration (isotonic): MAE={cal_mae:.6f}, Dir Acc={cal_dir:.1f}%")

    # --- Split OOS into optimization vs final holdout (when USE_FINAL_HOLDOUT) ---
    opt_indices, holdout_indices = None, None
    opt_preds, holdout_preds = None, None
    df_opt_5d, df_holdout_5d = None, None
    if USE_FINAL_HOLDOUT and len(oos_5d_indices) >= 50:
        n_oos = len(oos_5d_indices)
        order = sorted(range(n_oos), key=lambda i: oos_5d_indices[i])
        opt_n = max(1, int(n_oos * (1 - HOLDOUT_FRAC)))
        opt_pos, holdout_pos = order[:opt_n], order[opt_n:]
        opt_indices = [oos_5d_indices[i] for i in opt_pos]
        holdout_indices = [oos_5d_indices[i] for i in holdout_pos]
        opt_preds = (oos_5d_calibrated if (USE_CALIBRATION and calibrator_5d is not None) else oos_5d_preds)[opt_pos]
        holdout_preds = (oos_5d_calibrated if (USE_CALIBRATION and calibrator_5d is not None) else oos_5d_preds)[holdout_pos]
        df_opt_5d = df_oos_5d.loc[[i for i in opt_indices if i in df_oos_5d.index]]
        df_holdout_5d = df_oos_5d.loc[[i for i in holdout_indices if i in df_oos_5d.index]]
        print(f"Final holdout: {len(holdout_indices)} bars (last {HOLDOUT_FRAC*100:.0f}% of OOS); threshold/cap optimized on {len(opt_indices)} bars only.")

    # --- Re-optimize threshold and max size on calibrated OOS (or opt subset only when USE_FINAL_HOLDOUT) ---
    CALIBRATED_THRESHOLD = MIN_TRADE_THRESHOLD
    CALIBRATED_MAX_CAP = MAX_CONFIDENCE_CAP
    _thresh_indices = opt_indices if (USE_FINAL_HOLDOUT and opt_indices is not None) else oos_5d_indices
    _thresh_preds = opt_preds if (USE_FINAL_HOLDOUT and opt_preds is not None) else (oos_5d_calibrated if (USE_CALIBRATION and calibrator_5d is not None) else oos_5d_preds)
    _thresh_df = df_opt_5d if (USE_FINAL_HOLDOUT and df_opt_5d is not None) else df_oos_5d
    if USE_CALIBRATION and calibrator_5d is not None and len(_thresh_indices) >= 20:
        best_score = -np.inf
        best_thresh = MIN_TRADE_THRESHOLD
        best_cap = MAX_CONFIDENCE_CAP
        for thresh in [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035]:
            for cap in [2.0, 2.5, 3.0]:
                m = run_trade_sim(
                    _thresh_indices, _thresh_preds, _thresh_df, days=TRADE_WINDOW_DAYS,
                    min_threshold=thresh, max_conf_cap=cap, return_metrics=True,
                    execution_haircut_pct=EXECUTION_HAIRCUT_PCT)
                if m['n_trades'] < 10:
                    continue
                score = m['sharpe'] - 0.5 * m['max_drawdown']
                if score > best_score:
                    best_score = score
                    best_thresh = thresh
                    best_cap = cap
        CALIBRATED_THRESHOLD = best_thresh
        CALIBRATED_MAX_CAP = best_cap
        print(f"Optimized (calibrated) on {'holdout-opt subset' if USE_FINAL_HOLDOUT and opt_indices else 'full OOS'}: threshold={CALIBRATED_THRESHOLD*100:.2f}%, max_cap={CALIBRATED_MAX_CAP}")

    # --- Optional meta "trade vs no trade" gate (calibrated pred + vol regime + COT flags) ---
    trade_gate_proba = None
    trade_gate_clf = None
    if USE_TRADE_GATE and len(oos_5d_indices) >= 50:
        X_meta_list = []
        for i, idx in enumerate(oos_5d_indices):
            row = df_oos_5d.loc[idx] if idx in df_oos_5d.index else None
            if row is None:
                X_meta_list.append([oos_5d_calibrated[i], 0, 0, 0, 0])
                continue
            rv5_v = row.get('rv_5d_avg', row.get('realized_vol_5', 0)) or 0
            rv22_v = row.get('rv_22d_avg', row.get('realized_vol_20', 0)) or 0
            atr_v = row.get('atr_pct', 0) or 0
            X_meta_list.append([oos_5d_calibrated[i], rv5_v, rv22_v, atr_v, row.get('cot_gbp_Lev_Money_net_z', 0) or 0])
        X_meta = np.array(X_meta_list)
        y_meta = []
        for i, idx in enumerate(oos_5d_indices):
            row = df_oos_5d.loc[idx] if idx in df_oos_5d.index else None
            if row is None:
                y_meta.append(0)
                continue
            pred_ret = oos_5d_calibrated[i]
            if abs(pred_ret) < CALIBRATED_THRESHOLD:
                y_meta.append(0)
                continue
            surpass, profit, _, _ = check_surpass_and_pnl(row, df_oos_5d, pred_ret, days=TRADE_WINDOW_DAYS, trade_size=10000)
            y_meta.append(1 if profit > 0 else 0)
        y_meta = np.array(y_meta)
        if np.unique(y_meta).size >= 2:
            trade_gate_clf = LogisticRegression(max_iter=500, class_weight='balanced')
            trade_gate_clf.fit(X_meta, y_meta)
            trade_gate_proba = trade_gate_clf.predict_proba(X_meta)[:, 1]
            gate_acc = (trade_gate_proba >= 0.5).astype(int) == y_meta
            print(f"Trade gate (meta): OOS accuracy = {gate_acc.mean()*100:.1f}%")

    best_model_5d = xgb.XGBRegressor(objective='reg:squarederror', **last_5d_params)
    best_model_5d.fit(X_5d, y_5d)

    preds_for_sim = oos_5d_calibrated if (USE_CALIBRATION and calibrator_5d is not None) else oos_5d_preds
    thresh_sim = CALIBRATED_THRESHOLD if (USE_CALIBRATION and calibrator_5d is not None) else MIN_TRADE_THRESHOLD
    cap_sim = CALIBRATED_MAX_CAP if (USE_CALIBRATION and calibrator_5d is not None) else MAX_CONFIDENCE_CAP
    if EXECUTION_HAIRCUT_PCT > 0:
        print(f"Execution haircut applied: {EXECUTION_HAIRCUT_PCT*10000:.1f} bps round-trip.")
    for days in [5, 10, 15]:
        run_trade_sim(oos_5d_indices, preds_for_sim, df_oos_5d, days, label="5d-model ",
                     min_threshold=thresh_sim, max_conf_cap=cap_sim, trade_gate_proba=trade_gate_proba,
                     execution_haircut_pct=EXECUTION_HAIRCUT_PCT)

    # --- Holdout (unseen): report metrics on last HOLDOUT_FRAC of OOS only ---
    if USE_FINAL_HOLDOUT and holdout_indices is not None and len(holdout_indices) >= 10 and df_holdout_5d is not None:
        print("\n--- Holdout (unseen; threshold/cap not tuned on this period) ---")
        for days in [5, 10, 15]:
            run_trade_sim(holdout_indices, holdout_preds, df_holdout_5d, days, label="Holdout ",
                         min_threshold=thresh_sim, max_conf_cap=cap_sim,
                         execution_haircut_pct=EXECUTION_HAIRCUT_PCT)

    # --- A/B threshold scenarios (0.25%, 0.30%, 0.35%) for 5d window ---
    if RUN_AB_THRESHOLD_SCENARIOS and len(oos_5d_indices) > 0:
        print("\n--- A/B threshold scenarios (5d window, calibrated) ---")
        ab_thresholds = [0.0025, 0.003, 0.0035]
        ab_rows = []
        for th in ab_thresholds:
            m = run_trade_sim(oos_5d_indices, preds_for_sim, df_oos_5d, days=5,
                              min_threshold=th, max_conf_cap=cap_sim, trade_gate_proba=trade_gate_proba,
                              execution_haircut_pct=EXECUTION_HAIRCUT_PCT, return_metrics=True)
            ab_rows.append({
                'threshold_pct': th * 100,
                'n_trades': m['n_trades'],
                'surpass_pct': m['surpass_pct'],
                'account_pct': m['account_pct'],
                'sharpe': m['sharpe'],
                'max_drawdown': m['max_drawdown'] * 100,
            })
        print(f"  Threshold   Trades  Surpass%  Account%   Sharpe  MaxDD%")
        for r in ab_rows:
            print(f"  {r['threshold_pct']:.2f}%      {r['n_trades']:5d}   {r['surpass_pct']:5.1f}%   {r['account_pct']:+6.2f}%   {r['sharpe']:5.2f}   {r['max_drawdown']:.1f}%")

    # --- Time-slice analysis: early / mid / late third of OOS ---
    if RUN_TIME_SLICE_ANALYSIS and len(oos_5d_indices) > 0:
        print("\n--- Time-slice analysis (5d window, early/mid/late third) ---")
        df_oos_5d_sorted = df_oos_5d.loc[oos_5d_indices].sort_index()
        n = len(df_oos_5d_sorted)
        if n >= 3:
            third = n // 3
            slices = [
                ('early', df_oos_5d_sorted.index[:third].tolist()),
                ('mid', df_oos_5d_sorted.index[third:2 * third].tolist()),
                ('late', df_oos_5d_sorted.index[2 * third:].tolist()),
            ]
            print(f"  Slice   Trades  Surpass%  Account%   Sharpe  MaxDD%")
            for name, idx_list in slices:
                preds_slice = np.array([preds_for_sim[oos_5d_indices.index(i)] for i in idx_list if i in oos_5d_indices])
                # Build index list that matches preds_slice order
                idx_in_oos = [i for i in idx_list if i in oos_5d_indices]
                if len(idx_in_oos) < 5:
                    continue
                m = run_trade_sim(idx_in_oos, preds_slice, df_oos_5d.loc[idx_in_oos], days=5,
                                  min_threshold=thresh_sim, max_conf_cap=cap_sim,
                                  execution_haircut_pct=EXECUTION_HAIRCUT_PCT, return_metrics=True)
                print(f"  {name:5s}   {m['n_trades']:5d}   {m['surpass_pct']:5.1f}%   {m['account_pct']:+6.2f}%   {m['sharpe']:5.2f}   {m['max_drawdown']*100:.1f}%")

    # --- Calibration diagnostics: bin by vol regime (ATR% or rv_5d_avg tertiles) ---
    if RUN_CALIBRATION_DIAGNOSTICS and len(oos_5d_indices) >= 30 and calibrator_5d is not None:
        print("\n--- Calibration diagnostics by vol regime (tertiles of atr_pct) ---")
        vol_col = 'atr_pct' if 'atr_pct' in df_oos_5d.columns else 'rv_5d_avg'
        if vol_col not in df_oos_5d.columns:
            vol_col = 'realized_vol_5' if 'realized_vol_5' in df_oos_5d.columns else None
        if vol_col is not None:
            valid_idx = [i for i in oos_5d_indices if i in df_oos_5d.index]
            vol_vals = np.array([df_oos_5d.loc[i, vol_col] for i in valid_idx], dtype=float)
            pos_in_oos = [oos_5d_indices.index(i) for i in valid_idx]
            pred_cal = oos_5d_calibrated[pos_in_oos]
            actual = oos_5d_actuals[pos_in_oos]
            if len(vol_vals) >= 15:
                q1, q2 = np.nanpercentile(vol_vals, [33.33, 66.67])
                low = np.where(vol_vals <= q1)[0]
                mid = np.where((vol_vals > q1) & (vol_vals <= q2))[0]
                high = np.where(vol_vals > q2)[0]
                print(f"  Regime (by {vol_col})   N    Mean Pred%   Mean Real%   Bias%")
                for label, idx in [('low_vol', low), ('mid_vol', mid), ('high_vol', high)]:
                    if len(idx) < 5:
                        continue
                    mean_pred = np.mean(pred_cal[idx]) * 100
                    mean_real = np.mean(actual[idx]) * 100
                    bias = mean_pred - mean_real
                    print(f"  {label:10s}           {len(idx):4d}   {mean_pred:+.3f}      {mean_real:+.3f}      {bias:+.3f}")

    # ============================================================
    # Production prediction for latest day
    # ============================================================
    original_df = pd.read_csv('combined_GBPAUD_daily.csv', parse_dates=['date'])
    original_df, _ = load_and_merge_tff(original_df)
    original_df = compute_relative_features(original_df)
    # Pattern group counts for latest row (must match training)
    add_pattern_group_counts(original_df, pattern_cols)
    latest_row = original_df.iloc[-1]

    # Build feature vector in same order as X.columns (only columns that exist in original_df or are one-hot)
    latest_data = get_latest_feature_vector(X.columns, original_df, pattern_cols)

    current_close = latest_row['daily_close']
    pred_5d_raw = best_model_5d.predict(latest_data)[0]
    pred_5d = calibrator_5d.predict(np.array([pred_5d_raw]))[0] if (USE_CALIBRATION and calibrator_5d is not None) else pred_5d_raw

    prod_threshold = CALIBRATED_THRESHOLD if (USE_CALIBRATION and calibrator_5d is not None) else MIN_TRADE_THRESHOLD
    trade_ok = abs(pred_5d) > prod_threshold
    if USE_TRADE_GATE and trade_gate_clf is not None and trade_ok:
        rv5_v = latest_row.get('rv_5d_avg', latest_row.get('realized_vol_5', 0)) or 0
        rv22_v = latest_row.get('rv_22d_avg', latest_row.get('realized_vol_20', 0)) or 0
        atr_v = latest_row.get('atr_pct', 0) or 0
        cot_z_v = latest_row.get('cot_gbp_Lev_Money_net_z', 0) or 0
        X_meta_latest = np.array([[pred_5d, rv5_v, rv22_v, atr_v, cot_z_v]])
        gate_proba = trade_gate_clf.predict_proba(X_meta_latest)[0, 1]
        trade_ok = trade_ok and (gate_proba >= 0.5)

    print(f"\nCurrent Close: {current_close:.5f}")
    print(f"5-Day Predicted Return (raw): {pred_5d_raw*100:.4f}%")
    if USE_CALIBRATION and calibrator_5d is not None:
        print(f"5-Day Calibrated Return: {pred_5d*100:.4f}% -> {current_close*(1+pred_5d):.5f}")
    else:
        print(f"5-Day Predicted Return: {pred_5d*100:.4f}% -> {current_close*(1+pred_5d):.5f}")
    trade_signal = "LONG" if (trade_ok and pred_5d > prod_threshold) else ("SHORT" if (trade_ok and pred_5d < -prod_threshold) else "NO TRADE")
    print(f"Signal: {trade_signal} (threshold={prod_threshold*100:.2f}%)")


if __name__ == '__main__':
    main()
