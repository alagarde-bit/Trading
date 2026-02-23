import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from scipy.stats import mannwhitneyu

# Load and preprocess data
df = pd.read_csv('combined_GBPAUD_daily.csv', parse_dates=['date'])

# Simplify chart pattern labels
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

chart_pattern_cols = [
    'daily_chart_patterns', 'daily_chart_prev_pattern', 'daily_chart_prev_3_patterns', 'daily_chart_prev_10_patterns',
    'weekly_chart_patterns', 'weekly_chart_prev_pattern', 'weekly_chart_prev_3_patterns', 'weekly_chart_prev_10_patterns',
    'monthly_chart_patterns', 'monthly_chart_prev_pattern', 'monthly_chart_prev_3_patterns', 'monthly_chart_prev_10_patterns'
]
for col in chart_pattern_cols:
    df[col] = df[col].apply(simplify_chart_patterns)

# Create target
df['next_close'] = df['daily_close'].shift(-1)

# Features
num_features = [
    'daily_open', 'daily_high', 'daily_low', 'daily_close',
    'weekly_open', 'weekly_high', 'weekly_low', 'weekly_close',
    'monthly_open', 'monthly_high', 'monthly_low', 'monthly_close',
    'daily_sma_20', 'daily_sma_50', 'daily_sma_200', 'daily_rsi',
    'weekly_sma_4', 'weekly_sma_10', 'weekly_sma_40', 'weekly_rsi',
    'monthly_sma_1', 'monthly_sma_3', 'monthly_sma_10', 'monthly_rsi'
]
pattern_cols = [
    'daily_candlestick_patterns', 'daily_candlestick_prev_pattern', 'daily_candlestick_prev_3_patterns', 'daily_candlestick_prev_10_patterns',
    'daily_chart_patterns', 'daily_chart_prev_pattern', 'daily_chart_prev_3_patterns', 'daily_chart_prev_10_patterns',
    'weekly_candlestick_patterns', 'weekly_candlestick_prev_pattern', 'weekly_candlestick_prev_3_patterns', 'weekly_candlestick_prev_10_patterns',
    'weekly_chart_patterns', 'weekly_chart_prev_pattern', 'weekly_chart_prev_3_patterns', 'weekly_chart_prev_10_patterns',
    'monthly_candlestick_patterns', 'monthly_candlestick_prev_pattern', 'monthly_candlestick_prev_3_patterns', 'monthly_candlestick_prev_10_patterns',
    'monthly_chart_patterns', 'monthly_chart_prev_pattern', 'monthly_chart_prev_3_patterns', 'monthly_chart_prev_10_patterns'
]

# Encode patterns
encoded_dfs = []
for col in pattern_cols:
    mlb = MultiLabelBinarizer()
    df[col] = df[col].astype(str)
    patterns = df[col].str.split(';').fillna('')
    encoded = mlb.fit_transform(patterns)
    encoded_df = pd.DataFrame(encoded, columns=[f"{col}_{cls}" for cls in mlb.classes_], index=df.index)
    encoded_dfs.append(encoded_df)

# Combine features
X = pd.concat([df[num_features], pd.concat(encoded_dfs, axis=1)], axis=1)
y = df['next_close']

# Drop NaNs and align
valid_rows = X.notna().all(axis=1) & y.notna()
X = X.loc[valid_rows]
y = y.loc[valid_rows]
df = df.loc[valid_rows]

# Create holdout set (35%)
X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y, test_size=0.35, shuffle=False
)
df_holdout = df.loc[X_holdout.index]

# Define parameter grid
param_grid = {
    'n_estimators': [80, 100, 120],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [3, 4],
    'subsample': [0.9, 1.0]
}

# Initialize XGBoost model
model_close = xgb.XGBRegressor(objective='reg:squarederror')

# Time-series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Grid search
grid_search = GridSearchCV(
    estimator=model_close,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

# Best model and parameters
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation MAE: {-grid_search.best_score_:.4f}")

# Evaluate on holdout
pred_closes = best_model.predict(X_holdout)
mae_holdout = mean_absolute_error(y_holdout, pred_closes)
print(f"Holdout MAE: {mae_holdout:.4f}")

# Predict for the latest day
latest_data = X.iloc[-1:]
pred_close = best_model.predict(latest_data)[0]


# Compute residual threshold for risk management
train_preds = best_model.predict(X_train)
train_residuals = np.abs(train_preds - y_train)
residual_threshold = np.percentile(train_residuals, 85)
print(f"\nResidual Threshold (85th percentile): {residual_threshold:.4f}")

# Non-surpass analysis
def check_surpass(row, df, pred_close, days=5):
    current_open = row['daily_open']
    date_idx = row.name
    future_idx = df.index[df.index > date_idx][:days]
    if len(future_idx) == 0:
        return False
    
    future_highs = df.loc[future_idx, 'daily_high']
    future_lows = df.loc[future_idx, 'daily_low']
    future_closes = df.loc[future_idx, 'daily_close']
    
    if pred_close < current_open:  # Bearish prediction
        # Check if any low goes below predicted close OR any close is below predicted close
        return any(future_lows < pred_close) or any(future_closes < pred_close)
    else:  # Bullish prediction
        # Check if any high goes above predicted close OR any close is above predicted close
        return any(future_highs > pred_close) or any(future_closes > pred_close)

# Risk management: Identify high-confidence non-surpass
non_surpass_confident = []
for idx, pred_close in zip(X_holdout.index, pred_closes):
    row = df_holdout.loc[idx]
    residual = np.abs(pred_close - y_holdout.loc[idx])
    is_confident_non_surpass = residual > residual_threshold
    non_surpass_confident.append(is_confident_non_surpass)

# Label non-surpass
non_surpass = []
for idx, pred_close in zip(X_holdout.index, pred_closes):
    row = df_holdout.loc[idx]
    surpass_any = any(check_surpass(row, df_holdout, pred_close, days=d) for d in [1, 3, 5, 10, 15])
    non_surpass.append(0 if surpass_any else 1)
non_surpass = np.array(non_surpass)

# Opposite prediction analysis
opposite_days_non_surpass = []
opposite_days_surpass = []
max_streaks_non_surpass = []
max_streaks_surpass = []
threshold = 0.0025  # Minimum change to qualify as opposite (~0.125%)

for idx, pred_close in zip(X_holdout.index, pred_closes):
    row = df_holdout.loc[idx]
    trade_price = row['daily_close']
    initial_pred = pred_close
    is_bullish = pred_close > trade_price
    future_idx = df_holdout.index[df_holdout.index > idx][:15]
    if len(future_idx) == 0:
        continue
    
    # Get predicted closes for subsequent days
    future_pred_idx = np.searchsorted(X_holdout.index, future_idx, side='left')
    future_pred_idx = future_pred_idx[future_pred_idx < len(pred_closes)]
    future_preds = pred_closes[future_pred_idx]
    
    # Count opposite prediction days and streaks
    opposite_count = 0
    current_streak = 0
    max_streak = 0
    
    for current_pred in future_preds:
        if is_bullish:
            is_opposite = current_pred < initial_pred - threshold  # Below initial prediction
        else:
            is_opposite = current_pred > initial_pred + threshold  # Above initial prediction
        if is_opposite:
            opposite_count += 1
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    if non_surpass[X_holdout.index.get_loc(idx)]:
        opposite_days_non_surpass.append(opposite_count)
        max_streaks_non_surpass.append(max_streak)
    else:
        opposite_days_surpass.append(opposite_count)
        max_streaks_surpass.append(max_streak)

# Compute statistics
if opposite_days_non_surpass:
    print(f"\nOpposite Prediction Days Analysis (Non-Surpass):")
    print(f"Average Opposite Days: {np.mean(opposite_days_non_surpass):.2f}")
    print(f"Median Opposite Days: {np.median(opposite_days_non_surpass):.2f}")
    print(f"Average Consecutive Streak: {np.mean(max_streaks_non_surpass):.2f}")
    print(f"Median Consecutive Streak: {np.median(max_streaks_non_surpass):.2f}")
if opposite_days_surpass:
    print(f"\nOpposite Prediction Days Analysis (Surpass):")
    print(f"Average Opposite Days: {np.mean(opposite_days_surpass):.2f}")
    print(f"Median Opposite Days: {np.median(opposite_days_surpass):.2f}")
    print(f"Average Consecutive Streak: {np.mean(max_streaks_surpass):.2f}")
    print(f"Median Consecutive Streak: {np.median(max_streaks_surpass):.2f}")

# Mann-Whitney U test for differences
if opposite_days_non_surpass and opposite_days_surpass:
    stat, p_value = mannwhitneyu(opposite_days_non_surpass, opposite_days_surpass, alternative='two-sided')
    print(f"\nMann-Whitney U Test for Opposite Days:")
    print(f"P-value: {p_value:.4f}")
    stat, p_value = mannwhitneyu(max_streaks_non_surpass, max_streaks_surpass, alternative='two-sided')
    print(f"Mann-Whitney U Test for Consecutive Streaks:")
    print(f"P-value: {p_value:.4f}")

# Mann-Whitney U Test for RSI
rsi_non_surpass = df_holdout.loc[X_holdout.index, 'daily_rsi'][non_surpass == 1]
rsi_surpass = df_holdout.loc[X_holdout.index, 'daily_rsi'][non_surpass == 0]
if len(rsi_non_surpass) > 0 and len(rsi_surpass) > 0:
    stat, p_value = mannwhitneyu(rsi_non_surpass, rsi_surpass, alternative='two-sided')
    print(f"\nMann-Whitney U Test for Daily RSI:")
    print(f"Median RSI (Non-Surpass): {np.median(rsi_non_surpass):.2f}")
    print(f"Median RSI (Surpass): {np.median(rsi_surpass):.2f}")
    print(f"P-value: {p_value:.4f}")

# Mann-Whitney U Test for Weekly RSI
weekly_rsi_non_surpass = df_holdout.loc[X_holdout.index, 'weekly_rsi'][non_surpass == 1]
weekly_rsi_surpass = df_holdout.loc[X_holdout.index, 'weekly_rsi'][non_surpass == 0]
if len(weekly_rsi_non_surpass) > 0 and len(weekly_rsi_surpass) > 0:
    stat, p_value = mannwhitneyu(weekly_rsi_non_surpass, weekly_rsi_surpass, alternative='two-sided')
    print(f"\nMann-Whitney U Test for Weekly RSI:")
    print(f"Median Weekly RSI (Non-Surpass): {np.median(weekly_rsi_non_surpass):.2f}")
    print(f"Median Weekly RSI (Surpass): {np.median(weekly_rsi_surpass):.2f}")
    print(f"P-value: {p_value:.4f}")

# Mann-Whitney U Test for Monthly RSI
monthly_rsi_non_surpass = df_holdout.loc[X_holdout.index, 'monthly_rsi'][non_surpass == 1]
monthly_rsi_surpass = df_holdout.loc[X_holdout.index, 'monthly_rsi'][non_surpass == 0]
if len(monthly_rsi_non_surpass) > 0 and len(monthly_rsi_surpass) > 0:
    stat, p_value = mannwhitneyu(monthly_rsi_non_surpass, monthly_rsi_surpass, alternative='two-sided')
    print(f"\nMann-Whitney U Test for Monthly RSI:")
    print(f"Median Monthly RSI (Non-Surpass): {np.median(monthly_rsi_non_surpass):.2f}")
    print(f"Median Monthly RSI (Surpass): {np.median(monthly_rsi_surpass):.2f}")
    print(f"P-value: {p_value:.4f}")

# Mann-Whitney U Test for Weekly SMAs
weekly_sma_indicators = ['weekly_sma_4', 'weekly_sma_10', 'weekly_sma_40']
for sma in weekly_sma_indicators:
    sma_non_surpass = df_holdout.loc[X_holdout.index, sma][non_surpass == 1]
    sma_surpass = df_holdout.loc[X_holdout.index, sma][non_surpass == 0]
    if len(sma_non_surpass) > 0 and len(sma_surpass) > 0:
        stat, p_value = mannwhitneyu(sma_non_surpass, sma_surpass, alternative='two-sided')
        print(f"\nMann-Whitney U Test for {sma}:")
        print(f"Median {sma} (Non-Surpass): {np.median(sma_non_surpass):.4f}")
        print(f"Median {sma} (Surpass): {np.median(sma_surpass):.4f}")
        print(f"P-value: {p_value:.4f}")

# Mann-Whitney U Test for Monthly SMAs
monthly_sma_indicators = ['monthly_sma_1', 'monthly_sma_3', 'monthly_sma_10']
for sma in monthly_sma_indicators:
    sma_non_surpass = df_holdout.loc[X_holdout.index, sma][non_surpass == 1]
    sma_surpass = df_holdout.loc[X_holdout.index, sma][non_surpass == 0]
    if len(sma_non_surpass) > 0 and len(sma_surpass) > 0:
        stat, p_value = mannwhitneyu(sma_non_surpass, sma_surpass, alternative='two-sided')
        print(f"\nMann-Whitney U Test for {sma}:")
        print(f"Median {sma} (Non-Surpass): {np.median(sma_non_surpass):.4f}")
        print(f"Median {sma} (Surpass): {np.median(sma_surpass):.4f}")
        print(f"P-value: {p_value:.4f}")

# Variable Importance
clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
clf.fit(X_holdout, non_surpass)
importance = clf.get_booster().get_score(importance_type='gain')
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
print("\nTop 5 Features by Importance (Non-Surpass):")
for feature, score in sorted_importance[:5]:
    print(f"{feature}: {score:.4f}")

# Pattern Frequency
pattern_freq = {}
for col in pattern_cols:
    for idx in X_holdout.index:
        value = str(df_holdout.loc[idx, col]) if not pd.isna(df_holdout.loc[idx, col]) else ''
        patterns = value.split(';') if value else ['']
        for pattern in patterns:
            if pattern and pattern != '':
                key = f"{col}_{pattern}"
                if key not in pattern_freq:
                    pattern_freq[key] = {'non_surpass': 0, 'surpass': 0}
                pattern_freq[key]['non_surpass' if non_surpass[X_holdout.index.get_loc(idx)] else 'surpass'] += 1

print("\nTop 5 Patterns in Non-Surpass Cases:")
pattern_diff = [(k, v['non_surpass'] / (v['non_surpass'] + v['surpass'] + 1e-10)) 
                for k, v in pattern_freq.items() if v['non_surpass'] > 0]
pattern_diff.sort(key=lambda x: x[1], reverse=True)
for pattern, freq in pattern_diff[:5]:
    total = pattern_freq[pattern]['non_surpass'] + pattern_freq[pattern]['surpass']
    print(f"{pattern}: {freq*100:.2f}% of {total} occurrences")

# Model Confidence
residuals = np.abs(pred_closes - y_holdout)
margins = np.abs(pred_closes - df_holdout['daily_open'].values)
print("\nModel Confidence Metrics:")
print(f"Mean Residual (Non-Surpass): {residuals[non_surpass == 1].mean():.4f}")
print(f"Mean Residual (Surpass): {residuals[non_surpass == 0].mean():.4f}")
print(f"Mean Margin (Non-Surpass): {margins[non_surpass == 1].mean():.4f}")
print(f"Mean Margin (Surpass): {margins[non_surpass == 0].mean():.4f}")

# Distance from Moving Averages
for sma in ['daily_sma_20', 'daily_sma_50', 'daily_sma_200']:
    dist = np.abs(df_holdout['daily_close'] - df_holdout[sma]) / df_holdout['daily_close']
    dist_non_surpass = dist[X_holdout.index][non_surpass == 1]
    dist_surpass = dist[X_holdout.index][non_surpass == 0]
    if len(dist_non_surpass) > 0 and len(dist_surpass) > 0:
        stat, p_value = mannwhitneyu(dist_non_surpass, dist_surpass, alternative='two-sided')
        print(f"\nMann-Whitney U Test for Distance to {sma}:")
        print(f"Median Distance (Non-Surpass): {np.median(dist_non_surpass)*100:.2f}%")
        print(f"Median Distance (Surpass): {np.median(dist_surpass)*100:.2f}%")
        print(f"P-value: {p_value:.4f}")

# Trade performance with risk management
def check_surpass_and_pnl(row, df, pred_close, days=5, spread=0.0002, trade_size=10000):
    trade_price = row['daily_close']  # Previous day's close
    date_idx = row.name
    future_idx = df.index[df.index > date_idx][:days]
    if len(future_idx) == 0:
        return False, 0.0
    
    future_highs = df.loc[future_idx, 'daily_high']
    future_lows = df.loc[future_idx, 'daily_low']
    future_closes = df.loc[future_idx, 'daily_close']
    
    if pred_close < trade_price:  # Bearish
        # Check both lows and closes
        surpass = any(future_lows < pred_close) or any(future_closes < pred_close)
        entry_price = trade_price + spread
        # Check lows first for better profit
        for low in future_lows:
            if low < pred_close:
                ret = (entry_price - low - spread) / entry_price
                profit = ret * trade_size
                return surpass, profit
        # Then check closes
        for close in future_closes:
            if close < pred_close:
                ret = (entry_price - close - spread) / entry_price
                profit = ret * trade_size
                return surpass, profit
        exit_price = future_closes.iloc[-1] - spread
        ret = (entry_price - exit_price) / entry_price
        profit = ret * trade_size
        return surpass, profit
    else:  # Bullish
        # Check both highs and closes
        surpass = any(future_highs > pred_close) or any(future_closes > pred_close)
        entry_price = trade_price + spread
        # Check highs first for better profit
        for high in future_highs:
            if high > pred_close:
                ret = (high - entry_price - spread) / entry_price
                profit = ret * trade_size
                return surpass, profit
        # Then check closes
        for close in future_closes:
            if close > pred_close:
                ret = (close - entry_price - spread) / entry_price
                profit = ret * trade_size
                return surpass, profit
        exit_price = future_closes.iloc[-1] - spread
        ret = (exit_price - entry_price) / entry_price
        profit = ret * trade_size
        return surpass, profit

# Test window lengths
window_lengths = [1, 3, 5, 10, 15]
initial_balance = 100000  # Starting account balance
for days in window_lengths:
    surpass_results = []
    profits = []
    balance = initial_balance
    for idx, pred_close in zip(X_holdout.index, pred_closes):
        # Removed skip of high-confidence non-surpass trades
        row = df_holdout.loc[idx]
        surpass, profit = check_surpass_and_pnl(row, df_holdout, pred_close, days=days)
        surpass_results.append(surpass)
        profits.append(profit)
        balance += profit
    
    surpass_percentage = np.mean(surpass_results) * 100 if surpass_results else 0
    account_increase = ((balance - initial_balance) / initial_balance) * 100
    sharpe_ratio = np.mean(profits) / np.std(profits) * np.sqrt(252) if profits and np.std(profits) != 0 else 0
    print(f"\nWindow: {days} days")
    print(f"Surpass Percentage: {surpass_percentage:.2f}%")
    print(f"Account Increase: {account_increase:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")



# Get the original latest data (2025-06-17) that was dropped during training
original_df = pd.read_csv('combined_GBPAUD_daily.csv', parse_dates=['date'])
latest_row = original_df.iloc[-1]  # This is 2025-06-17 with close 2.07249

# Process the latest row the same way as training data
latest_features = []
for col in num_features:
    latest_features.append(latest_row[col])

# Process patterns for the latest row using the same encoding as training
pattern_idx = 0
for col in pattern_cols:
    # Get the encoded features for this pattern column
    pattern_features = [f for f in X.columns if f.startswith(col + '_')]
    
    # Get the latest row's patterns
    latest_patterns = str(latest_row[col]).split(';') if pd.notna(latest_row[col]) else ['']
    
    # Create a dictionary of pattern presence
    pattern_dict = {pattern: 1 for pattern in latest_patterns if pattern}
    
    # Add encoded features (1 if pattern exists, 0 otherwise)
    for feature in pattern_features:
        pattern_name = feature.replace(col + '_', '')
        latest_features.append(pattern_dict.get(pattern_name, 0))

# Create feature array for prediction
latest_data = np.array(latest_features).reshape(1, -1)

# Make prediction
pred_close = best_model.predict(latest_data)[0]
current_close = latest_row['daily_close']
print(f"Current Close: {current_close:.5f}")
print(f"Predicted Close: {pred_close:.5f}")
print(f"Predicted Change: {(pred_close - current_close):.5f}")