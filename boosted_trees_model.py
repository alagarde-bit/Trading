import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# Load the dataset
df = pd.read_csv('combined_GBPAUD_daily.csv', parse_dates=['date'])

# Function to simplify chart pattern labels
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

# Simplify chart pattern columns
chart_pattern_cols = ['chart_patterns', 'chart_prev_pattern', 
                      'chart_prev_3_patterns', 'chart_prev_10_patterns']
for col in chart_pattern_cols:
    df[col] = df[col].apply(simplify_chart_patterns)

# Create target variables
df['next_high'] = df['high'].shift(-1)
df['next_low'] = df['low'].shift(-1)
df['next_close'] = df['close'].shift(-1)

# Drop rows with NaN in targets
df = df.dropna(subset=['next_high', 'next_low', 'next_close'])

# Define feature columns
num_features = ['open', 'high', 'low', 'close', 'sma_20', 'sma_50', 'sma_200', 'rsi']
pattern_cols = ['candlestick_patterns', 'candlestick_prev_pattern', 
                'candlestick_prev_3_patterns', 'candlestick_prev_10_patterns',
                'chart_patterns', 'chart_prev_pattern', 
                'chart_prev_3_patterns', 'chart_prev_10_patterns']

# Encode pattern columns
encoded_dfs = []
for col in pattern_cols:
    mlb = MultiLabelBinarizer()
    patterns = df[col].str.split(';').fillna('')
    encoded = mlb.fit_transform(patterns)
    encoded_df = pd.DataFrame(encoded, columns=[f"{col}_{cls}" for cls in mlb.classes_], 
                            index=df.index)
    encoded_dfs.append(encoded_df)

# Combine features
X = pd.concat([df[num_features], pd.concat(encoded_dfs, axis=1)], axis=1)
y_high = df['next_high']
y_low = df['next_low']
y_close = df['next_close']

# Drop rows with NaN in features and align targets
X = X.dropna()
y_high = y_high.loc[X.index]
y_low = y_low.loc[X.index]
y_close = y_close.loc[X.index]

# Time series split
tscv = TimeSeriesSplit(n_splits=5)

# Initialize XGBoost models (following gradient boosting steps)
model_high = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, 
                             objective='reg:squarederror')
model_low = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, 
                            objective='reg:squarederror')
model_close = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, 
                              objective='reg:squarederror')

# Train and evaluate using the last split
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_high, y_test_high = y_high.iloc[train_idx], y_high.iloc[test_idx]
    y_train_low, y_test_low = y_low.iloc[train_idx], y_low.iloc[test_idx]
    y_train_close, y_test_close = y_close.iloc[train_idx], y_close.iloc[test_idx]

    # Fit models
    model_high.fit(X_train, y_train_high)
    model_low.fit(X_train, y_train_low)
    model_close.fit(X_train, y_train_close)

    # Predict and evaluate
    pred_high = model_high.predict(X_test)
    pred_low = model_low.predict(X_test)
    pred_close = model_close.predict(X_test)

    mae_high = mean_absolute_error(y_test_high, pred_high)
    mae_low = mean_absolute_error(y_test_low, pred_low)
    mae_close = mean_absolute_error(y_test_close, pred_close)
    print(f"MAE - High: {mae_high:.4f}, Low: {mae_low:.4f}, Close: {mae_close:.4f}")

# Predict for the latest day
latest_data = X.iloc[-1:]
pred_high = model_high.predict(latest_data)[0]
pred_low = model_low.predict(latest_data)[0]
pred_close = model_close.predict(latest_data)[0]
print(f"Predicted next day - High: {pred_high:.4f}, Low: {pred_low:.4f}, Close: {pred_close:.4f}")