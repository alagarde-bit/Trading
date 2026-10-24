import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from scipy.stats import mannwhitneyu

# --- TFF (CFTC Commitments of Traders) integration ---
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
IMPORTANCE_CV_MAX = 2.0            # Drop features with (std/mean) importance > this (unstable)
N_WF_FOLDS = 5
MIN_TRAIN_FRAC = 0.50
DIRECTION_THRESHOLD = 0.001
USE_DIRECTION_CLASSIFIER = False
MIN_TRADE_THRESHOLD = 0.002
STOP_LOSS_MULT = 1.5
BASE_TRADE_SIZE = 10000

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

def load_and_merge_tff(daily_df, tff_path=TFF_PATH, commodity_filter=None):
    if commodity_filter is None:
        commodity_filter = ['BRITISH POUND', 'AUSTRALIAN DOLLAR']
    if not Path(tff_path).exists():
        return daily_df, []

    tff = pd.read_csv(tff_path)
    name_col = 'CONTRACT_MARKET_NAME' if 'CONTRACT_MARKET_NAME' in tff.columns else 'Market_and_Exchange_Names'
    if name_col == 'Market_and_Exchange_Names':
        tff['_contract'] = tff[name_col].str.extract(r'^([A-Z\s]+)\s*-', expand=False).str.strip()
    else:
        tff['_contract'] = tff[name_col].astype(str).str.strip()
    date_col = 'Report_Date_as_YYYY_MM_DD'
    tff[date_col] = pd.to_datetime(tff[date_col], format='%Y %b %d %I:%M:%S %p', errors='coerce')
    tff = tff.dropna(subset=[date_col])

    out_df = daily_df.copy()
    cot_feature_names = []
    derived_by_label = {}

    for label, commodity in [('gbp', 'BRITISH POUND'), ('aud', 'AUSTRALIAN DOLLAR')]:
        if commodity not in commodity_filter:
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
        feat_cols = [c for c in derived.columns if c != 'report_date']
        cot_feature_names.extend(feat_cols)
        daily_sorted = out_df.sort_values('date')
        merged = pd.merge_asof(daily_sorted[['date']].copy(), derived, left_on='date', right_on='report_date', direction='backward')
        for c in feat_cols:
            out_df[c] = out_df['date'].map(merged.set_index('date')[c])
        out_df[f'cot_{label}_available'] = (out_df[feat_cols[0]].notna()).astype(np.float64)
        cot_feature_names.append(f'cot_{label}_available')

    if 'gbp' in derived_by_label and 'aud' in derived_by_label:
        for g in ['Dealer', 'Asset_Mgr', 'Lev_Money']:
            gbp_net, aud_net = f'cot_gbp_{g}_net', f'cot_aud_{g}_net'
            if gbp_net in out_df.columns and aud_net in out_df.columns:
                out_df[f'cot_diff_{g}_net'] = out_df[gbp_net] - out_df[aud_net]
                cot_feature_names.append(f'cot_diff_{g}_net')
                gbp_chg, aud_chg = f'cot_gbp_{g}_net_chg', f'cot_aud_{g}_net_chg'
                if gbp_chg in out_df.columns and aud_chg in out_df.columns:
                    out_df[f'cot_diff_{g}_net_chg'] = out_df[gbp_chg] - out_df[aud_chg]
                    cot_feature_names.append(f'cot_diff_{g}_net_chg')

        # LevMoney − AssetMgr spread per currency, then GBP−AUD; 4w/8w change and z-scores
        gbp_lev, gbp_am = 'cot_gbp_Lev_Money_net', 'cot_gbp_Asset_Mgr_net'
        aud_lev, aud_am = 'cot_aud_Lev_Money_net', 'cot_aud_Asset_Mgr_net'
        if all(c in out_df.columns for c in [gbp_lev, gbp_am, aud_lev, aud_am]):
            out_df['cot_gbp_lev_minus_am'] = out_df[gbp_lev] - out_df[gbp_am]
            out_df['cot_aud_lev_minus_am'] = out_df[aud_lev] - out_df[aud_am]
            out_df['cot_diff_lev_minus_am'] = out_df['cot_gbp_lev_minus_am'] - out_df['cot_aud_lev_minus_am']
            for col in ['cot_gbp_lev_minus_am', 'cot_aud_lev_minus_am', 'cot_diff_lev_minus_am']:
                out_df[f'{col}_4w_chg'] = out_df[col] - out_df[col].shift(4 * 5)
                out_df[f'{col}_8w_chg'] = out_df[col] - out_df[col].shift(8 * 5)
                roll_mean = out_df[col].rolling(COT_ZSCORE_WINDOW, min_periods=4).mean()
                roll_std = out_df[col].rolling(COT_ZSCORE_WINDOW, min_periods=4).std()
                out_df[f'{col}_z'] = (out_df[col] - roll_mean) / roll_std.replace(0, np.nan)
                cot_feature_names.extend([col, f'{col}_4w_chg', f'{col}_8w_chg', f'{col}_z'])

    # COT multi-week momentum and 52-week percentile (extremes signal)
    def _pct_rank_52w(s):
        if s.isna().all() or len(s) < 52:
            return np.nan
        return (s.iloc[-1] >= s).mean() * 100

    for label in ['gbp', 'aud']:
        for g in ['Dealer', 'Asset_Mgr', 'Lev_Money']:
            col = f'cot_{label}_{g}_net'
            if col not in out_df.columns:
                continue
            out_df[f'{col}_4w_chg'] = out_df[col] - out_df[col].shift(4 * 5)
            out_df[f'{col}_8w_chg'] = out_df[col] - out_df[col].shift(8 * 5)
            out_df[f'{col}_momentum'] = out_df[f'{col}_4w_chg'] - out_df[f'{col}_8w_chg']
            out_df[f'{col}_52w_pct'] = out_df[col].rolling(260, min_periods=52).apply(_pct_rank_52w, raw=False)
            cot_feature_names.extend([f'{col}_4w_chg', f'{col}_8w_chg', f'{col}_momentum', f'{col}_52w_pct'])

    for c in cot_feature_names:
        if c in out_df.columns and not c.endswith('_available') and out_df[c].isna().any():
            out_df[c] = out_df[c].fillna(0.0)
    return out_df, cot_feature_names

def compute_relative_features(df):
    """Replace raw prices with stationary relative features."""
    df['daily_return'] = df['daily_close'].pct_change()
    df['daily_range_pct'] = (df['daily_high'] - df['daily_low']) / df['daily_close']
    df['daily_body_pct'] = (df['daily_close'] - df['daily_open']) / df['daily_close']
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

    # Position pain: COT z × recent return (crowded-and-wrong)
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
    return df

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df, cot_features = load_and_merge_tff(df)
    df = compute_relative_features(df)

    if cot_features and TRAIN_COT_ONLY:
        dropped_avail = 0
        avail_cols = [c for c in cot_features if c.endswith('_available')]
        if avail_cols:
            before = len(df)
            df = df.loc[df[avail_cols].all(axis=1)].copy()
            dropped_avail = before - len(df)
        if COT_START_DATE is not None:
            start = pd.to_datetime(COT_START_DATE)
            before = len(df)
            df = df.loc[df['date'] >= start].copy()
            dropped_date = before - len(df)
            print(f"COT-only mode: from {COT_START_DATE}, using {len(df)} rows (dropped {dropped_avail} without COT, {dropped_date} before start date)")
        elif avail_cols:
            print(f"COT-only mode: using {len(df)} rows (dropped {dropped_avail} without COT)")

    weekly_cols = [col for col in df.columns if col.startswith('weekly_')]
    df_weekly = df.drop_duplicates(subset=weekly_cols).copy()
    monthly_cols = [col for col in df.columns if col.startswith('monthly_')]
    df_monthly = df.drop_duplicates(subset=monthly_cols).copy()

    # Return target instead of price level; direction class for optional classifier
    df_weekly['next_close_raw'] = df_weekly['weekly_close'].shift(-1)
    df_weekly['next_return'] = df_weekly['next_close_raw'] / df_weekly['weekly_close'] - 1
    df_weekly['target_class'] = np.where(
        df_weekly['next_return'] > DIRECTION_THRESHOLD, 1,
        np.where(df_weekly['next_return'] < -DIRECTION_THRESHOLD, -1, 0)
    )
    df_monthly['next_close_raw'] = df_monthly['monthly_close'].shift(-1)
    df_monthly['next_return'] = df_monthly['next_close_raw'] / df_monthly['monthly_close'] - 1
    df_monthly['target_class'] = np.where(
        df_monthly['next_return'] > DIRECTION_THRESHOLD, 1,
        np.where(df_monthly['next_return'] < -DIRECTION_THRESHOLD, -1, 0)
    )

    return df_weekly, df_monthly, cot_features

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

# Aligned with actual pattern names in data: candlestick as-is, chart after simplify (First_Last).
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

def _pattern_to_group(name, group_map=PATTERN_GROUP_MAP):
    if not name or not isinstance(name, str):
        return None
    n = name.lower().strip()
    for group, keywords in group_map.items():
        if any(kw in n for kw in keywords):
            return group
    return None

def add_pattern_group_counts_df(df, pattern_cols, group_map=PATTERN_GROUP_MAP):
    pattern_grp_cols = [f'pattern_grp_{g}' for g in group_map]
    def count_groups(row):
        cnt = {g: 0 for g in group_map}
        for col in pattern_cols:
            if col not in row.index:
                continue
            for part in str(row[col]).split(';'):
                grp = _pattern_to_group(part, group_map)
                if grp:
                    cnt[grp] += 1
        return pd.Series({f'pattern_grp_{g}': cnt[g] for g in group_map})
    df[pattern_grp_cols] = df.apply(count_groups, axis=1)
    return df

def prepare_features(df, pattern_cols, cot_features=None):
    num_features = [
        'daily_return', 'daily_range_pct', 'daily_body_pct',
        'close_vs_daily_sma20', 'close_vs_daily_sma50', 'close_vs_daily_sma200',
        'daily_sma20_vs_sma50', 'daily_sma50_vs_sma200',
        'daily_rsi',
        'atr_pct', 'bb_width', 'realized_vol_5', 'realized_vol_20', 'vol_ratio',
        'roc_5', 'roc_20', 'macd', 'macd_signal', 'macd_hist',
        'daily_rsi_chg', 'weekly_rsi_chg', 'monthly_rsi_chg',
        'atr_pct_chg', 'bb_width_chg', 'realized_vol_5_chg', 'realized_vol_20_chg',
        'close_vs_sma20_chg', 'close_vs_sma50_chg', 'close_vs_sma200_chg',
        'daily_rsi_acc', 'atr_pct_acc',
        'return_5d_avg', 'return_22d_avg', 'realized_vol_22',
        'rv_5d_avg', 'rv_22d_avg', 'atr_pct_5d_avg', 'atr_pct_22d_avg',
        'area_sma20_10d', 'area_sma20_20d', 'area_sma50_10d', 'area_sma50_20d',
        'area_sma200_10d', 'area_sma200_20d',
        'return_autocorr_10d', 'return_autocorr_20d',
        'return_sq_autocorr_10d', 'return_sq_autocorr_20d',
        'return_skew_10d', 'return_skew_20d', 'return_kurt_10d', 'return_kurt_20d',
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
    num_features = [f for f in num_features if f in df.columns]
    if cot_features:
        num_features = num_features + [f for f in cot_features if f in df.columns]
    _cot_extra = [c for c in df.columns if c.startswith('cot_') and not c.endswith('_available') and c not in num_features]
    num_features = num_features + _cot_extra

    add_pattern_group_counts_df(df, pattern_cols)
    pattern_grp_cols = [f'pattern_grp_{g}' for g in PATTERN_GROUP_MAP]
    num_features = num_features + [c for c in pattern_grp_cols if c in df.columns]

    encoded_cols = []
    all_encoded = None
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
        all_encoded = all_encoded.loc[:, keep_mask]
        encoded_cols = list(all_encoded.columns)
        kept_pat = int(keep_mask.sum())
        dropped_pat = int((~keep_mask).sum())
        print(f"Pattern features: {kept_pat} kept, {dropped_pat} dropped (< {MIN_PATTERN_OCC} occurrences)")
    else:
        print("Pattern features: using group counts + strength only")

    if all_encoded is not None:
        X = pd.concat([df[num_features], all_encoded], axis=1)
    else:
        X = df[num_features].copy()
    y = df['next_return']

    valid_rows = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]
    df = df.loc[valid_rows]

    return X, y, df, num_features

def build_interaction_constraints(feature_names, num_features_set):
    """COT <-> trend/vol only; pattern <-> trend/vol only. Returns list of lists of feature names (XGBoost expects names when X is DataFrame)."""
    cot_names = [n for n in feature_names if n.startswith('cot_') and not n.endswith('_available')]
    pattern_names = [n for n in feature_names if n.startswith('pattern_grp_') or (n not in num_features_set)]
    trend_vol_names = [n for n in feature_names if n not in cot_names and n not in pattern_names]
    if not cot_names or not trend_vol_names:
        return None
    return [cot_names + trend_vol_names, trend_vol_names + pattern_names]

# --- Surpass functions (return-based) ---
def check_surpass(row, df_ref, pred_return, timeframe, periods):
    tf = timeframe.lower()
    current_close = row[f'{tf}_close']
    pred_close = current_close * (1 + pred_return)
    date_idx = row.name
    future_idx = df_ref.index[df_ref.index > date_idx][:periods]
    if len(future_idx) == 0:
        return False

    future_highs = df_ref.loc[future_idx, f'{tf}_high']
    future_lows = df_ref.loc[future_idx, f'{tf}_low']
    future_closes = df_ref.loc[future_idx, f'{tf}_close']

    if pred_return < 0:
        return any(future_lows < pred_close) or any(future_closes < pred_close)
    else:
        return any(future_highs > pred_close) or any(future_closes > pred_close)

def check_surpass_and_pnl(row, df_ref, pred_return, timeframe, periods,
                          spread=0.0002, trade_size=10000, stop_loss_mult=STOP_LOSS_MULT):
    """Close-based exit with stop-loss. Returns (surpassed, pnl, exit_reason)."""
    tf = timeframe.lower()
    entry_price = row[f'{tf}_close']
    pred_close = entry_price * (1 + pred_return)
    stop_dist = abs(pred_return) * stop_loss_mult
    date_idx = row.name
    future_idx = df_ref.index[df_ref.index > date_idx][:periods]
    if len(future_idx) == 0:
        return False, 0.0, 'no_data'

    future_closes = df_ref.loc[future_idx, f'{tf}_close']
    is_long = pred_return > 0

    for close in future_closes:
        ret = (close - entry_price) / entry_price
        trade_ret = ret if is_long else -ret

        if trade_ret <= -stop_dist:
            pnl = (-stop_dist - spread / entry_price) * trade_size
            return False, pnl, 'stop'

        if is_long and close >= pred_close:
            pnl = (ret - spread / entry_price) * trade_size
            return True, pnl, 'target'
        elif not is_long and close <= pred_close:
            pnl = (-ret - spread / entry_price) * trade_size
            return True, pnl, 'target'

    final_close = future_closes.iloc[-1]
    ret = (final_close - entry_price) / entry_price
    pnl = (ret if is_long else -ret) * trade_size - (spread / entry_price) * trade_size
    surpass = (final_close >= pred_close) if is_long else (final_close <= pred_close)
    return surpass, pnl, 'expire'

def analyze_timeframe(df, X, y, timeframe, pattern_cols, periods, num_features):
    # --- Walk-forward expanding-window validation ---
    n = len(X)
    min_train = int(n * MIN_TRAIN_FRAC)
    test_chunk = max(1, (n - min_train) // N_WF_FOLDS)

    param_grid = {
        'n_estimators': [80, 100, 120],
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth': [3, 4],
        'subsample': [0.9, 1.0]
    }

    all_oos_indices = []
    all_oos_preds = []
    all_oos_actuals = []
    last_best_params = None
    fold_importances = []

    num_features_set = set(num_features)
    interaction_constraints = None
    if USE_INTERACTION_CONSTRAINTS:
        interaction_constraints = build_interaction_constraints(list(X.columns), num_features_set)
        if interaction_constraints is not None:
            print(f"{timeframe} interaction constraints: COT<->trend/vol, pattern<->trend/vol")

    print(f"\n--- {timeframe} Walk-Forward ({N_WF_FOLDS} folds, min train {MIN_TRAIN_FRAC*100:.0f}%) ---")
    print(f"Total rows: {n}, min train: {min_train}, test chunk: ~{test_chunk}")

    for fold in range(N_WF_FOLDS):
        train_end = min_train + fold * test_chunk
        test_end = train_end + test_chunk if fold < N_WF_FOLDS - 1 else n

        X_train_fold = X.iloc[:train_end]
        y_train_fold = y.iloc[:train_end]
        X_test_fold = X.iloc[train_end:test_end]
        y_test_fold = y.iloc[train_end:test_end]

        if len(X_test_fold) == 0:
            continue

        model_kw = dict(objective='reg:squarederror')
        if interaction_constraints is not None:
            model_kw['interaction_constraints'] = interaction_constraints
        model = xgb.XGBRegressor(**model_kw)
        n_inner = min(3, len(X_train_fold) // 2)
        if n_inner < 2:
            n_inner = 2
        tscv_inner = TimeSeriesSplit(n_splits=n_inner)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=tscv_inner,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train_fold, y_train_fold)
        last_best_params = grid_search.best_params_
        fold_importances.append(grid_search.best_estimator_.feature_importances_)

        preds = grid_search.best_estimator_.predict(X_test_fold)
        fold_mae = mean_absolute_error(y_test_fold, preds)
        actual_signs = np.sign(y_test_fold.values)
        pred_signs = np.sign(preds)
        fold_dir_acc = np.mean(pred_signs == actual_signs) * 100

        print(f"  Fold {fold+1}: Train={train_end}, Test={len(X_test_fold)}, "
              f"MAE={fold_mae:.6f}, Dir Acc={fold_dir_acc:.1f}%")

        all_oos_indices.extend(X_test_fold.index.tolist())
        all_oos_preds.extend(preds.tolist())
        all_oos_actuals.extend(y_test_fold.values.tolist())

    all_oos_preds = np.array(all_oos_preds)
    all_oos_actuals = np.array(all_oos_actuals)

    oos_mae = mean_absolute_error(all_oos_actuals, all_oos_preds)
    oos_dir_acc = np.mean(np.sign(all_oos_preds) == np.sign(all_oos_actuals)) * 100
    print(f"\n{timeframe} Aggregate OOS Return MAE: {oos_mae:.6f}")
    print(f"{timeframe} Aggregate OOS Directional Accuracy: {oos_dir_acc:.1f}%")

    # Feature importance stability across folds; optionally drop low/unstable features
    if fold_importances and len(fold_importances) == N_WF_FOLDS:
        imp_arr = np.array(fold_importances)
        imp_mean = imp_arr.mean(axis=0)
        imp_std = imp_arr.std(axis=0)
        imp_cv = np.where(imp_mean > 1e-8, imp_std / imp_mean, np.nan)
        low_thresh = np.nanpercentile(imp_mean, IMPORTANCE_LOW_PCT)
        drop_low = imp_mean < low_thresh
        drop_unstable = np.isfinite(imp_cv) & (imp_cv > IMPORTANCE_CV_MAX)
        drop_mask = drop_low | drop_unstable
        stable_names = [X.columns[i] for i in range(len(drop_mask)) if not drop_mask[i]]
        unstable_or_low = [X.columns[i] for i in range(len(drop_mask)) if drop_mask[i]]
        if unstable_or_low and DROP_UNSTABLE_FEATURES:
            print(f"{timeframe} feature importance: dropping {len(unstable_or_low)} low/unstable (low_pct={IMPORTANCE_LOW_PCT}, cv_max={IMPORTANCE_CV_MAX})")
            X = X[stable_names]

    # Train final production model on all data (use current X columns after optional drop)
    num_features_current = set(c for c in num_features if c in X.columns)
    ic_final = build_interaction_constraints(list(X.columns), num_features_current) if USE_INTERACTION_CONSTRAINTS else None
    model_final_kw = dict(objective='reg:squarederror', **last_best_params)
    if ic_final is not None:
        model_final_kw['interaction_constraints'] = ic_final
    best_model = xgb.XGBRegressor(**model_final_kw)
    best_model.fit(X, y)
    print(f"{timeframe} final model trained on all {len(X)} rows with params: {last_best_params}")

    if USE_DIRECTION_CLASSIFIER and 'target_class' in df.columns:
        y_class = df['target_class'].astype(int)
        split_idx = max(1, int(len(X) * 0.80))
        X_clf_train, X_clf_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_clf_train, y_clf_test = y_class.iloc[:split_idx], y_class.iloc[split_idx:]
        clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, use_label_encoder=False, eval_metric='mlogloss')
        clf.fit(X_clf_train, y_clf_train)
        clf_acc = (clf.predict(X_clf_test) == y_clf_test.values).mean() * 100
        print(f"{timeframe} direction classifier OOS class accuracy = {clf_acc:.1f}% (threshold={DIRECTION_THRESHOLD})")

    # --- OOS analysis ---
    df_oos = df.loc[all_oos_indices]
    X_oos = X.loc[all_oos_indices]

    oos_residuals = np.abs(all_oos_preds - all_oos_actuals)
    residual_threshold = np.percentile(oos_residuals, 85)
    print(f"\n{timeframe} Residual Threshold (85th pctl of OOS): {residual_threshold:.6f}")

    non_surpass_confident = []
    for i in range(len(all_oos_indices)):
        non_surpass_confident.append(oos_residuals[i] > residual_threshold)

    # --- Trade performance (realistic) ---
    median_pred = np.median(np.abs(all_oos_preds))
    for period in periods:
        initial_balance = 100000
        balance = initial_balance
        trades_taken = 0
        results = {'long': {'surpass': 0, 'count': 0, 'pnl': 0.0, 'profits': []},
                   'short': {'surpass': 0, 'count': 0, 'pnl': 0.0, 'profits': []}}
        exit_reasons = {'target': 0, 'stop': 0, 'expire': 0, 'no_data': 0}
        all_profits = []

        for i, idx in enumerate(all_oos_indices):
            pred_ret = all_oos_preds[i]
            if abs(pred_ret) < MIN_TRADE_THRESHOLD:
                continue

            row = df_oos.loc[idx]
            confidence = abs(pred_ret) / median_pred if median_pred > 0 else 1.0
            trade_size = BASE_TRADE_SIZE * min(confidence, 3.0)

            surpass, profit, reason = check_surpass_and_pnl(
                row, df_oos, pred_ret, timeframe, period, trade_size=trade_size)

            direction = 'long' if pred_ret > 0 else 'short'
            results[direction]['count'] += 1
            results[direction]['surpass'] += int(surpass)
            results[direction]['pnl'] += profit
            results[direction]['profits'].append(profit)
            all_profits.append(profit)
            balance += profit
            trades_taken += 1
            exit_reasons[reason] += 1

        total_surpass = results['long']['surpass'] + results['short']['surpass']
        surpass_pct = (total_surpass / trades_taken * 100) if trades_taken > 0 else 0
        acct_inc = ((balance - initial_balance) / initial_balance) * 100
        annualize = 52 if timeframe == 'Weekly' else 12
        sharpe = (np.mean(all_profits) / np.std(all_profits) * np.sqrt(annualize)
                  if all_profits and np.std(all_profits) > 0 else 0)

        print(f"\n{'='*60}")
        print(f"{timeframe} Window: {period} periods | Trades: {trades_taken}/{len(all_oos_indices)} "
              f"(filtered {len(all_oos_indices) - trades_taken})")
        print(f"  Surpass: {surpass_pct:.1f}% | Account: {acct_inc:+.2f}% | Sharpe: {sharpe:.2f}")
        print(f"  Exits: target={exit_reasons['target']}, stop={exit_reasons['stop']}, "
              f"expire={exit_reasons['expire']}")

        for d in ['long', 'short']:
            r = results[d]
            if r['count'] == 0:
                continue
            d_surp = r['surpass'] / r['count'] * 100
            d_sharpe = (np.mean(r['profits']) / np.std(r['profits']) * np.sqrt(annualize)
                        if r['profits'] and np.std(r['profits']) > 0 else 0)
            print(f"  {d.upper():5s}: {r['count']} trades, surpass {d_surp:.1f}%, "
                  f"PnL {r['pnl']:+.2f}, Sharpe {d_sharpe:.2f}")

    # --- Non-surpass analysis ---
    for period in periods:
        non_surpass = []
        for i, idx in enumerate(all_oos_indices):
            row = df_oos.loc[idx]
            pred_ret = all_oos_preds[i]
            surpass = check_surpass(row, df_oos, pred_ret, timeframe, period)
            non_surpass.append(0 if surpass else 1)
        non_surpass = np.array(non_surpass)

        print(f"\n{timeframe} Non-Surpass Analysis for {period} periods:")
        print(f"Non-Surpass Percentage: {(non_surpass.mean() * 100):.2f}%")

        rsi_col = f'{timeframe.lower()}_rsi'
        rsi_ns = df_oos[rsi_col].values[non_surpass == 1]
        rsi_s = df_oos[rsi_col].values[non_surpass == 0]
        if len(rsi_ns) > 0 and len(rsi_s) > 0:
            stat, p_value = mannwhitneyu(rsi_ns, rsi_s, alternative='two-sided')
            print(f"Mann-Whitney U Test for {timeframe} RSI:")
            print(f"Median RSI (Non-Surpass): {np.median(rsi_ns):.2f}")
            print(f"Median RSI (Surpass): {np.median(rsi_s):.2f}")
            print(f"P-value: {p_value:.4f}")

        pattern_freq = {}
        for col in pattern_cols:
            for i, idx in enumerate(all_oos_indices):
                value = str(df_oos.loc[idx, col]) if not pd.isna(df_oos.loc[idx, col]) else ''
                pats = value.split(';') if value else ['']
                for pattern in pats:
                    if pattern and pattern != '':
                        key = f"{col}_{pattern}"
                        if key not in pattern_freq:
                            pattern_freq[key] = {'non_surpass': 0, 'surpass': 0}
                        pattern_freq[key]['non_surpass' if non_surpass[i] else 'surpass'] += 1

        print(f"\nTop 5 Patterns in Non-Surpass Cases ({timeframe}, {period} periods):")
        pattern_diff = [(k, v['non_surpass'] / (v['non_surpass'] + v['surpass'] + 1e-10))
                        for k, v in pattern_freq.items() if v['non_surpass'] > 0]
        pattern_diff.sort(key=lambda x: x[1], reverse=True)
        for pattern, freq in pattern_diff[:5]:
            total = pattern_freq[pattern]['non_surpass'] + pattern_freq[pattern]['surpass']
            print(f"{pattern}: {freq*100:.2f}% of {total} occurrences")

    # --- Variable Importance ---
    if len(np.unique(non_surpass)) > 1:
        clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
        clf.fit(X_oos, non_surpass)
        importance = clf.get_booster().get_score(importance_type='gain')
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        print(f"\nTop 5 Features by Importance ({timeframe}):")
        for feature, score in sorted_importance[:5]:
            print(f"{feature}: {score:.4f}")

    # --- Production prediction ---
    print(f"\nPrediction for Next {timeframe}:")

    original_df = pd.read_csv('combined_GBPAUD_daily.csv', parse_dates=['date'])
    original_df, _ = load_and_merge_tff(original_df)
    original_df = compute_relative_features(original_df)
    latest_row = original_df.iloc[-1]

    tf = timeframe.lower()
    current_close = latest_row[f'{tf}_close']
    print(f"Current {timeframe} Close: {current_close:.5f}")

    latest_features = []
    for col in num_features:
        latest_features.append(latest_row[col])

    for col in pattern_cols:
        pattern_features = [f for f in X.columns if f.startswith(col + '_')]
        latest_patterns = str(latest_row[col]).split(';') if pd.notna(latest_row[col]) else ['']
        pattern_dict = {pattern: 1 for pattern in latest_patterns if pattern}
        for feature in pattern_features:
            pattern_name = feature.replace(col + '_', '')
            latest_features.append(pattern_dict.get(pattern_name, 0))

    latest_data = np.array(latest_features).reshape(1, -1)

    pred_return = best_model.predict(latest_data)[0]
    pred_close = current_close * (1 + pred_return)
    print(f"Predicted Return: {pred_return*100:.4f}%")
    print(f"Predicted Next {timeframe} Close: {pred_close:.5f}")
    print(f"Predicted Change: {(pred_close - current_close):.5f}")

    return best_model, X, y

def main():
    df_weekly, df_monthly, cot_features = load_and_preprocess_data('combined_GBPAUD_daily.csv')

    pattern_cols = [
        'daily_candlestick_patterns', 'daily_candlestick_prev_pattern', 'daily_candlestick_prev_3_patterns', 'daily_candlestick_prev_10_patterns',
        'daily_chart_patterns', 'daily_chart_prev_pattern', 'daily_chart_prev_3_patterns', 'daily_chart_prev_10_patterns',
        'weekly_candlestick_patterns', 'weekly_candlestick_prev_pattern', 'weekly_candlestick_prev_3_patterns', 'weekly_candlestick_prev_10_patterns',
        'weekly_chart_patterns', 'weekly_chart_prev_pattern', 'weekly_chart_prev_3_patterns', 'weekly_chart_prev_10_patterns',
        'monthly_candlestick_patterns', 'monthly_candlestick_prev_pattern', 'monthly_candlestick_prev_3_patterns', 'monthly_candlestick_prev_10_patterns',
        'monthly_chart_patterns', 'monthly_chart_prev_pattern', 'monthly_chart_prev_3_patterns', 'monthly_chart_prev_10_patterns'
    ]

    chart_pattern_cols = [col for col in pattern_cols if 'chart' in col]
    for df in [df_weekly, df_monthly]:
        for col in chart_pattern_cols:
            df[col] = df[col].apply(simplify_chart_patterns)

    X_weekly, y_weekly, df_weekly, num_features_w = prepare_features(df_weekly, pattern_cols, cot_features)
    weekly_periods = [1, 2, 4, 8, 12]
    best_model_weekly, X_weekly, y_weekly = analyze_timeframe(
        df_weekly, X_weekly, y_weekly, "Weekly", pattern_cols, weekly_periods, num_features_w)

    X_monthly, y_monthly, df_monthly, num_features_m = prepare_features(df_monthly, pattern_cols, cot_features)
    monthly_periods = [1, 2, 3, 6, 12]
    best_model_monthly, X_monthly, y_monthly = analyze_timeframe(
        df_monthly, X_monthly, y_monthly, "Monthly", pattern_cols, monthly_periods, num_features_m)

if __name__ == "__main__":
    main()
