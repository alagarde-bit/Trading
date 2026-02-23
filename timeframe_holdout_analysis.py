import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from scipy.stats import mannwhitneyu

def load_and_preprocess_data(file_path):
    # Load data
    df = pd.read_csv(file_path, parse_dates=['date'])
    
    # Create weekly and monthly dataframes
    df_weekly = df.copy()
    df_monthly = df.copy()
    
    # Filter for distinct weekly values
    weekly_cols = [col for col in df.columns if col.startswith('weekly_')]
    df_weekly = df_weekly.drop_duplicates(subset=weekly_cols)
    
    # Filter for distinct monthly values
    monthly_cols = [col for col in df.columns if col.startswith('monthly_')]
    df_monthly = df_monthly.drop_duplicates(subset=monthly_cols)
    
    # Create targets
    df_weekly['next_close'] = df_weekly['weekly_close'].shift(-1)
    df_monthly['next_close'] = df_monthly['monthly_close'].shift(-1)
    
    return df_weekly, df_monthly

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

def prepare_features(df, pattern_cols):
    # Numerical features
    num_features = [
        'daily_open', 'daily_high', 'daily_low', 'daily_close',
        'weekly_open', 'weekly_high', 'weekly_low', 'weekly_close',
        'monthly_open', 'monthly_high', 'monthly_low', 'monthly_close',
        'daily_sma_20', 'daily_sma_50', 'daily_sma_200', 'daily_rsi',
        'weekly_sma_4', 'weekly_sma_10', 'weekly_sma_40', 'weekly_rsi',
        'monthly_sma_1', 'monthly_sma_3', 'monthly_sma_10', 'monthly_rsi'
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
    
    return X, y, df

def check_surpass(row, df, pred_close, timeframe, periods):
    current_open = row[f'{timeframe.lower()}_open']
    date_idx = row.name
    future_idx = df.index[df.index > date_idx][:periods]
    if len(future_idx) == 0:
        return False
    
    future_highs = df.loc[future_idx, f'{timeframe.lower()}_high']
    future_lows = df.loc[future_idx, f'{timeframe.lower()}_low']
    future_closes = df.loc[future_idx, f'{timeframe.lower()}_close']
    
    if pred_close < current_open:  # Bearish prediction
        # Check if any low goes below predicted close OR any close is below predicted close
        return any(future_lows < pred_close) or any(future_closes < pred_close)
    else:  # Bullish prediction
        # Check if any high goes above predicted close OR any close is above predicted close
        return any(future_highs > pred_close) or any(future_closes > pred_close)

def check_surpass_and_pnl(row, df, pred_close, timeframe, periods, spread=0.0002, trade_size=10000):
    trade_price = row[f'{timeframe.lower()}_close']  # Previous period's close
    date_idx = row.name
    future_idx = df.index[df.index > date_idx][:periods]
    if len(future_idx) == 0:
        return False, 0.0
    
    future_highs = df.loc[future_idx, f'{timeframe.lower()}_high']
    future_lows = df.loc[future_idx, f'{timeframe.lower()}_low']
    future_closes = df.loc[future_idx, f'{timeframe.lower()}_close']
    
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

def analyze_timeframe(df, X, y, timeframe, pattern_cols, periods):
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
    model = xgb.XGBRegressor(objective='reg:squarederror')
    
    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    
    print(f"\nTraining {timeframe} Model...")
    grid_search.fit(X_train, y_train)
    
    # Best model and parameters
    best_model = grid_search.best_estimator_
    print(f"\n{timeframe} Model Best Parameters: {grid_search.best_params_}")
    print(f"{timeframe} Model Best Cross-Validation MAE: {-grid_search.best_score_:.4f}")
    
    # Evaluate on holdout
    pred_closes = best_model.predict(X_holdout)
    mae_holdout = mean_absolute_error(y_holdout, pred_closes)
    print(f"{timeframe} Model Holdout MAE: {mae_holdout:.4f}")
    
    # Compute residual threshold for risk management
    train_preds = best_model.predict(X_train)
    train_residuals = np.abs(train_preds - y_train)
    residual_threshold = np.percentile(train_residuals, 85)
    print(f"\n{timeframe} Residual Threshold (85th percentile): {residual_threshold:.4f}")
    
    # Risk management: Identify high-confidence non-surpass
    non_surpass_confident = []
    for idx, pred_close in zip(X_holdout.index, pred_closes):
        residual = np.abs(pred_close - y_holdout.loc[idx])
        is_confident_non_surpass = residual > residual_threshold
        non_surpass_confident.append(is_confident_non_surpass)
    
    # Test window lengths
    initial_balance = 100000  # Starting account balance
    for period in periods:
        surpass_results = []
        profits = []
        balance = initial_balance
        for idx, pred_close in zip(X_holdout.index, pred_closes):
            # Removed skip of high-confidence non-surpass trades
            row = df_holdout.loc[idx]
            surpass, profit = check_surpass_and_pnl(row, df_holdout, pred_close, timeframe, period)
            surpass_results.append(surpass)
            profits.append(profit)
            balance += profit
        
        surpass_percentage = np.mean(surpass_results) * 100 if surpass_results else 0
        account_increase = ((balance - initial_balance) / initial_balance) * 100
        sharpe_ratio = np.mean(profits) / np.std(profits) * np.sqrt(252) if profits and np.std(profits) != 0 else 0
        print(f"\n{timeframe} Window: {period} periods")
        print(f"Surpass Percentage: {surpass_percentage:.2f}%")
        print(f"Account Increase: {account_increase:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # Non-surpass analysis for different periods
    for period in periods:
        non_surpass = []
        for idx, pred_close in zip(X_holdout.index, pred_closes):
            row = df_holdout.loc[idx]
            surpass = check_surpass(row, df_holdout, pred_close, timeframe, period)
            non_surpass.append(0 if surpass else 1)
        non_surpass = np.array(non_surpass)
        
        print(f"\n{timeframe} Non-Surpass Analysis for {period} periods:")
        print(f"Non-Surpass Percentage: {(non_surpass.mean() * 100):.2f}%")
        
        # Mann-Whitney U Test for RSI
        rsi_col = f'{timeframe.lower()}_rsi'
        rsi_non_surpass = df_holdout.loc[X_holdout.index, rsi_col][non_surpass == 1]
        rsi_surpass = df_holdout.loc[X_holdout.index, rsi_col][non_surpass == 0]
        if len(rsi_non_surpass) > 0 and len(rsi_surpass) > 0:
            stat, p_value = mannwhitneyu(rsi_non_surpass, rsi_surpass, alternative='two-sided')
            print(f"Mann-Whitney U Test for {timeframe} RSI:")
            print(f"Median RSI (Non-Surpass): {np.median(rsi_non_surpass):.2f}")
            print(f"Median RSI (Surpass): {np.median(rsi_surpass):.2f}")
            print(f"P-value: {p_value:.4f}")
        
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
        
        print(f"\nTop 5 Patterns in Non-Surpass Cases ({timeframe}, {period} periods):")
        pattern_diff = [(k, v['non_surpass'] / (v['non_surpass'] + v['surpass'] + 1e-10)) 
                        for k, v in pattern_freq.items() if v['non_surpass'] > 0]
        pattern_diff.sort(key=lambda x: x[1], reverse=True)
        for pattern, freq in pattern_diff[:5]:
            total = pattern_freq[pattern]['non_surpass'] + pattern_freq[pattern]['surpass']
            print(f"{pattern}: {freq*100:.2f}% of {total} occurrences")
    
    # Variable Importance
    clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    clf.fit(X_holdout, non_surpass)
    importance = clf.get_booster().get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 5 Features by Importance ({timeframe}):")
    for feature, score in sorted_importance[:5]:
        print(f"{feature}: {score:.4f}")
    
    # Production prediction for next period
    print(f"\nPrediction for Next {timeframe}:")
    
    # Get the original latest data that was dropped during training
    original_df = pd.read_csv('combined_GBPAUD_daily.csv', parse_dates=['date'])
    
    # For weekly/monthly, we need to get the latest complete period
    if timeframe == "Weekly":
        # Get latest weekly data
        latest_row = original_df.iloc[-1]  # Latest daily data
        current_close = latest_row['weekly_close']
        print(f"Current {timeframe} Close: {current_close:.5f}")
        
        # Process the latest row the same way as training data
        latest_features = []
        num_features = [
            'daily_open', 'daily_high', 'daily_low', 'daily_close',
            'weekly_open', 'weekly_high', 'weekly_low', 'weekly_close',
            'monthly_open', 'monthly_high', 'monthly_low', 'monthly_close',
            'daily_sma_20', 'daily_sma_50', 'daily_sma_200', 'daily_rsi',
            'weekly_sma_4', 'weekly_sma_10', 'weekly_sma_40', 'weekly_rsi',
            'monthly_sma_1', 'monthly_sma_3', 'monthly_sma_10', 'monthly_rsi'
        ]
        
        for col in num_features:
            latest_features.append(latest_row[col])
        
        # Process patterns for the latest row using the same encoding as training
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
        print(f"Predicted Next {timeframe} Close: {pred_close:.5f}")
        print(f"Predicted Change: {(pred_close - current_close):.5f}")
        
    elif timeframe == "Monthly":
        # Get latest monthly data
        latest_row = original_df.iloc[-1]  # Latest daily data
        current_close = latest_row['monthly_close']
        print(f"Current {timeframe} Close: {current_close:.5f}")
        
        # Process the latest row the same way as training data
        latest_features = []
        num_features = [
            'daily_open', 'daily_high', 'daily_low', 'daily_close',
            'weekly_open', 'weekly_high', 'weekly_low', 'weekly_close',
            'monthly_open', 'monthly_high', 'monthly_low', 'monthly_close',
            'daily_sma_20', 'daily_sma_50', 'daily_sma_200', 'daily_rsi',
            'weekly_sma_4', 'weekly_sma_10', 'weekly_sma_40', 'weekly_rsi',
            'monthly_sma_1', 'monthly_sma_3', 'monthly_sma_10', 'monthly_rsi'
        ]
        
        for col in num_features:
            latest_features.append(latest_row[col])
        
        # Process patterns for the latest row using the same encoding as training
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
        print(f"Predicted Next {timeframe} Close: {pred_close:.5f}")
        print(f"Predicted Change: {(pred_close - current_close):.5f}")
    
    return best_model, X, y

def main():
    # Load and preprocess data
    df_weekly, df_monthly = load_and_preprocess_data('combined_GBPAUD_daily.csv')
    
    # Pattern columns
    pattern_cols = [
        'daily_candlestick_patterns', 'daily_candlestick_prev_pattern', 'daily_candlestick_prev_3_patterns', 'daily_candlestick_prev_10_patterns',
        'daily_chart_patterns', 'daily_chart_prev_pattern', 'daily_chart_prev_3_patterns', 'daily_chart_prev_10_patterns',
        'weekly_candlestick_patterns', 'weekly_candlestick_prev_pattern', 'weekly_candlestick_prev_3_patterns', 'weekly_candlestick_prev_10_patterns',
        'weekly_chart_patterns', 'weekly_chart_prev_pattern', 'weekly_chart_prev_3_patterns', 'weekly_chart_prev_10_patterns',
        'monthly_candlestick_patterns', 'monthly_candlestick_prev_pattern', 'monthly_candlestick_prev_3_patterns', 'monthly_candlestick_prev_10_patterns',
        'monthly_chart_patterns', 'monthly_chart_prev_pattern', 'monthly_chart_prev_3_patterns', 'monthly_chart_prev_10_patterns'
    ]
    
    # Simplify chart patterns
    chart_pattern_cols = [col for col in pattern_cols if 'chart' in col]
    for df in [df_weekly, df_monthly]:
        for col in chart_pattern_cols:
            df[col] = df[col].apply(simplify_chart_patterns)
    
    # Analyze weekly timeframe
    X_weekly, y_weekly, df_weekly = prepare_features(df_weekly, pattern_cols)
    weekly_periods = [1, 2, 4, 8, 12]  # weeks
    best_model_weekly, X_weekly, y_weekly = analyze_timeframe(df_weekly, X_weekly, y_weekly, "Weekly", pattern_cols, weekly_periods)
    
    # Analyze monthly timeframe
    X_monthly, y_monthly, df_monthly = prepare_features(df_monthly, pattern_cols)
    monthly_periods = [1, 2, 3, 6, 12]  # months
    best_model_monthly, X_monthly, y_monthly = analyze_timeframe(df_monthly, X_monthly, y_monthly, "Monthly", pattern_cols, monthly_periods)

if __name__ == "__main__":
    main() 