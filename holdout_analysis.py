import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from scipy.stats import mannwhitneyu, chi2_contingency

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

chart_pattern_cols = ['chart_patterns', 'chart_prev_pattern', 
                     'chart_prev_3_patterns', 'chart_prev_10_patterns']
for col in chart_pattern_cols:
    df[col] = df[col].apply(simplify_chart_patterns)

# Create targets
df['next_close'] = df['close'].shift(-1)

# Features
num_features = ['open', 'high', 'low', 'close', 'sma_20', 'sma_50', 'sma_200', 'rsi']
pattern_cols = ['candlestick_patterns', 'candlestick_prev_pattern', 
                'candlestick_prev_3_patterns', 'candlestick_prev_10_patterns',
                'chart_patterns', 'chart_prev_pattern', 
                'chart_prev_3_patterns', 'chart_prev_10_patterns']

# Encode patterns
encoded_dfs = []
for col in pattern_cols:
    mlb = MultiLabelBinarizer()
    patterns = df[col].str.split(';').fillna('')
    encoded = mlb.fit_transform(patterns)
    encoded_df = pd.DataFrame(encoded, columns=[f"{col}_{cls}" for cls in mlb.classes_], 
                             index=df.index)
    encoded_dfs.append(encoded_df)

X = pd.concat([df[num_features], pd.concat(encoded_dfs, axis=1)], axis=1)
y = df['next_close']

# Drop NaNs in features and targets, then align
valid_rows = X.notna().all(axis=1) & y.notna()
X = X.loc[valid_rows]
y = y.loc[valid_rows]
df = df.loc[valid_rows]

# Create holdout set (last 20%)
X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
df_holdout = df.loc[X_holdout.index]

# Define refined parameter grid
param_grid = {
    'n_estimators': [80, 100, 120],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [3, 4],
    'subsample': [0.9, 1.0]
}

# Initialize XGBoost model
model_close = xgb.XGBRegressor(objective='reg:squarederror')

# Set up time-series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Perform grid search
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

# Evaluate best model on holdout set
pred_closes = best_model.predict(X_holdout)
mae_holdout = mean_absolute_error(y_holdout, pred_closes)
print(f"Holdout MAE: {mae_holdout:.4f}")

# Non-surpass analysis
def check_surpass(row, df, pred_close, days=5):
    current_open = row['open']
    date_idx = row.name
    future_idx = df.index[df.index > date_idx][:days]
    if len(future_idx) == 0:
        return False
    future_closes = df.loc[future_idx, 'close']
    if pred_close < current_open:
        return any(future_closes < pred_close)
    else:
        return any(future_closes > pred_close)

# Label non-surpass
non_surpass = []
for idx, pred_close in zip(X_holdout.index, pred_closes):
    row = df_holdout.loc[idx]
    # Non-surpass if no surpass in any window
    surpass_any = any(check_surpass(row, df_holdout, pred_close, days=d) for d in [1, 3, 5, 10, 15])
    non_surpass.append(0 if surpass_any else 1)
non_surpass = np.array(non_surpass)

# 1. Mann-Whitney U Test for RSI
rsi_non_surpass = df_holdout.loc[X_holdout.index, 'rsi'][non_surpass == 1]
rsi_surpass = df_holdout.loc[X_holdout.index, 'rsi'][non_surpass == 0]
if len(rsi_non_surpass) > 0 and len(rsi_surpass) > 0:
    stat, p_value = mannwhitneyu(rsi_non_surpass, rsi_surpass, alternative='two-sided')
    print(f"\nMann-Whitney U Test for RSI:")
    print(f"Median RSI (Non-Surpass): {np.median(rsi_non_surpass):.2f}")
    print(f"Median RSI (Surpass): {np.median(rsi_surpass):.2f}")
    print(f"P-value: {p_value:.4f}")
else:
    print("\nMann-Whitney U Test for RSI: Insufficient data for test.")

# 2. Variable Importance
clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
clf.fit(X_holdout, non_surpass)
importance = clf.get_booster().get_score(importance_type='gain')
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
print("\nTop 5 Features by Importance (Non-Surpass):")
for feature, score in sorted_importance[:5]:
    print(f"{feature}: {score:.4f}")

# 3. Chart Pattern and Candlestick Labels
pattern_freq = {}
for col in pattern_cols:
    for idx in X_holdout.index:
        # Ensure value is string, handle NaN or non-string
        value = str(df_holdout.loc[idx, col]) if not pd.isna(df_holdout.loc[idx, col]) else ''
        patterns = value.split(';') if value else ['']
        for pattern in patterns:
            if pattern and pattern != '':  # Skip empty patterns
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
# 4. Model Confidence
residuals = np.abs(pred_closes - y_holdout)
margins = np.abs(pred_closes - df_holdout['open'].values)
print("\nModel Confidence Metrics:")
print(f"Mean Residual (Non-Surpass): {residuals[non_surpass == 1].mean():.4f}")
print(f"Mean Residual (Surpass): {residuals[non_surpass == 0].mean():.4f}")
print(f"Mean Margin (Non-Surpass): {margins[non_surpass == 1].mean():.4f}")
print(f"Mean Margin (Surpass): {margins[non_surpass == 0].mean():.4f}")

# 5. Distance from Moving Averages
for sma in ['sma_20', 'sma_50', 'sma_200']:
    dist = np.abs(df_holdout['close'] - df_holdout[sma]) / df_holdout['close']
    dist_non_surpass = dist[X_holdout.index][non_surpass == 1]
    dist_surpass = dist[X_holdout.index][non_surpass == 0]
    if len(dist_non_surpass) > 0 and len(dist_surpass) > 0:
        stat, p_value = mannwhitneyu(dist_non_surpass, dist_surpass, alternative='two-sided')
        print(f"\nMann-Whitney U Test for Distance to {sma}:")
        print(f"Median Distance (Non-Surpass): {np.median(dist_non_surpass)*100:.2f}%")
        print(f"Median Distance (Surpass): {np.median(dist_surpass)*100:.2f}%")
        print(f"P-value: {p_value:.4f}")
    else:
        print(f"\nMann-Whitney U Test for Distance to {sma}: Insufficient data for test.")

# Holdout analysis for multiple windows
def check_surpass_and_pnl(row, df, pred_close, days=5, spread=0.0002, max_return=0.1):
    current_open = row['open']
    date_idx = row.name
    future_idx = df.index[df.index > date_idx][:days]
    if len(future_idx) == 0:
        return False, 0.0
    future_closes = df.loc[future_idx, 'close']
    
    if pred_close < current_open:  # Bearish
        surpass = any(future_closes < pred_close)
        entry_price = current_open + spread
        for close in future_closes:
            if close < pred_close:
                ret = (entry_price - (close - spread)) / entry_price
                return surpass, min(max_return, max(-max_return, ret))
        exit_price = future_closes.iloc[-1] - spread
        ret = (entry_price - exit_price) / entry_price
        return surpass, min(max_return, max(-max_return, ret))
    else:  # Bullish
        surpass = any(future_closes > pred_close)
        entry_price = current_open + spread
        for close in future_closes:
            if close > pred_close:
                ret = ((close - spread) - entry_price) / entry_price
                return surpass, min(max_return, max(-max_return, ret))
        exit_price = future_closes.iloc[-1] - spread
        ret = (exit_price - entry_price) / entry_price
        return surpass, min(max_return, max(-max_return, ret))

# Test window lengths
window_lengths = [1, 3, 5, 10, 15]
for days in window_lengths:
    surpass_results = []
    returns = []
    for idx, pred_close in zip(X_holdout.index, pred_closes):
        row = df_holdout.loc[idx]
        surpass, ret = check_surpass_and_pnl(row, df_holdout, pred_close, days=days)
        surpass_results.append(surpass)
        returns.append(ret)
    
    surpass_percentage = np.mean(surpass_results) * 100
    cumulative_return = (np.prod([1 + r for r in returns]) - 1) * 100
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
    print(f"\nWindow: {days} days")
    print(f"Surpass Percentage: {surpass_percentage:.2f}%")
    print(f"Cumulative Return: {cumulative_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")