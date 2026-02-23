import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

class NZDJPYSurpassAnalysisWithPatterns:
    def __init__(self, csv_file, model_file):
        self.csv_file = csv_file
        self.model_file = model_file
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.pattern_encoders = {}
        self.sequence_length = 60
        self.df = None
        self.feature_df = None
        
    def load_data_and_model(self):
        """Load the data and trained model"""
        print("Loading data and model...")
        
        # Load data
        self.df = pd.read_csv(self.csv_file)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        # Select features (same as the training model)
        pattern_columns = [
            'daily_candlestick_patterns', 'daily_candlestick_prev_10_patterns',
            'daily_chart_patterns', 'daily_chart_prev_10_patterns',
            'weekly_candlestick_patterns', 'monthly_candlestick_patterns',
            'monthly_candlestick_prev_10_patterns', 'monthly_chart_patterns',
            'monthly_chart_prev_10_patterns'
        ]
        
        feature_columns = [
            'daily_open', 'daily_high', 'daily_low', 'daily_close',
            'weekly_open', 'weekly_high', 'weekly_low', 'weekly_close',
            'monthly_open', 'monthly_high', 'monthly_low', 'monthly_close',
            'daily_sma_20', 'daily_sma_50', 'daily_sma_200', 'daily_rsi',
            'weekly_sma_4', 'weekly_sma_10', 'weekly_sma_40', 'weekly_rsi',
            'monthly_sma_1', 'monthly_sma_3', 'monthly_sma_10', 'monthly_rsi'
        ] + pattern_columns
        
        self.feature_df = self.df[['date'] + feature_columns].dropna()
        
        # Load the trained model
        self.model = load_model(self.model_file)
        
        print(f"Data loaded: {len(self.feature_df)} records")
        print("Model loaded successfully")
        
    def prepare_scalers(self):
        """Prepare scalers using training data (up to 2025-09-05)"""
        cutoff_date = pd.to_datetime('2025-09-05')
        train_data = self.feature_df[self.feature_df['date'] <= cutoff_date].copy()
        
        # Encode pattern columns first
        train_data_encoded = self.encode_patterns(train_data)
        
        # Prepare features and targets for scaling
        feature_columns = [col for col in train_data_encoded.columns if col not in ['date', 'daily_close', 'weekly_close', 'monthly_close']]
        target_columns = ['daily_close', 'weekly_close', 'monthly_close']
        
        features = train_data_encoded[feature_columns].values
        targets = train_data_encoded[target_columns].values
        
        # Fit scalers
        self.feature_scaler.fit(features)
        self.scaler.fit(targets)
        
        print("Scalers fitted on training data")
        
    def encode_patterns(self, data):
        """Encode pattern columns using label encoding"""
        pattern_columns = [
            'daily_candlestick_patterns', 'daily_candlestick_prev_10_patterns',
            'daily_chart_patterns', 'daily_chart_prev_10_patterns',
            'weekly_candlestick_patterns', 'monthly_candlestick_patterns',
            'monthly_candlestick_prev_10_patterns', 'monthly_chart_patterns',
            'monthly_chart_prev_10_patterns'
        ]
        
        data_encoded = data.copy()
        
        for col in pattern_columns:
            if col in data_encoded.columns:
                # Fill NaN values with 'None' for pattern columns
                data_encoded[col] = data_encoded[col].fillna('None')
                
                # Create or use existing encoder
                if col not in self.pattern_encoders:
                    self.pattern_encoders[col] = LabelEncoder()
                    # Fit on all unique values in the column
                    unique_values = data_encoded[col].astype(str).unique()
                    self.pattern_encoders[col].fit(unique_values)
                
                # Transform the column
                data_encoded[col] = self.pattern_encoders[col].transform(data_encoded[col].astype(str))
        
        return data_encoded
        
    def create_sequences(self, data, target_columns):
        """Create sequences for prediction"""
        # Encode pattern columns
        data_encoded = self.encode_patterns(data)
        
        feature_columns = [col for col in data_encoded.columns if col not in ['date'] + target_columns]
        features = data_encoded[feature_columns].values
        features_scaled = self.feature_scaler.transform(features)
        
        X = []
        for i in range(self.sequence_length, len(features_scaled)):
            X.append(features_scaled[i-self.sequence_length:i])
        
        return np.array(X)
    
    def make_predictions(self, start_idx, end_idx):
        """Make predictions for a given range"""
        target_columns = ['daily_close', 'weekly_close', 'monthly_close']
        
        # Get data range
        data_range = self.feature_df.iloc[start_idx:end_idx].copy()
        
        if len(data_range) < self.sequence_length:
            return None, None
        
        # Create sequences
        X = self.create_sequences(data_range, target_columns)
        
        if len(X) == 0:
            return None, None
        
        # Make predictions
        predictions_scaled = self.model.predict(X, verbose=0)
        predictions = self.scaler.inverse_transform(predictions_scaled)
        
        # Get actual values for comparison
        actual_data = data_range.iloc[self.sequence_length:].copy()
        
        return predictions, actual_data
    
    def calculate_surpass_percentage(self, predictions, actual_data, target_col, high_col, days_ahead):
        """Calculate surpass percentage for a specific target and time horizon"""
        if predictions is None or actual_data is None:
            return None
        
        if len(predictions) < days_ahead:
            return None
        
        surpass_count = 0
        total_count = 0
        
        for i in range(len(predictions) - days_ahead + 1):
            # Get prediction
            predicted_close = predictions[i][0]  # daily_close is first target
            
            # Get actual values for the next 'days_ahead' days
            future_data = actual_data.iloc[i:i+days_ahead]
            
            if len(future_data) < days_ahead:
                continue
            
            # Check if predicted close is surpassed by any close or high in the period
            max_close = future_data['daily_close'].max()
            max_high = future_data['daily_high'].max()
            max_value = max(max_close, max_high)
            
            if max_value > predicted_close:
                surpass_count += 1
            
            total_count += 1
        
        if total_count == 0:
            return None
        
        return (surpass_count / total_count) * 100
    
    def calculate_weekly_surpass_percentage(self, predictions, actual_data, days_ahead):
        """Calculate surpass percentage for weekly predictions"""
        if predictions is None or actual_data is None:
            return None
        
        if len(predictions) < days_ahead:
            return None
        
        surpass_count = 0
        total_count = 0
        
        for i in range(len(predictions) - days_ahead + 1):
            # Get weekly prediction (second target)
            predicted_weekly_close = predictions[i][1]
            
            # Get actual weekly values for the next 'days_ahead' days
            future_data = actual_data.iloc[i:i+days_ahead]
            
            if len(future_data) < days_ahead:
                continue
            
            # Check if predicted weekly close is surpassed
            max_weekly_close = future_data['weekly_close'].max()
            max_weekly_high = future_data['weekly_high'].max()
            max_value = max(max_weekly_close, max_weekly_high)
            
            if max_value > predicted_weekly_close:
                surpass_count += 1
            
            total_count += 1
        
        if total_count == 0:
            return None
        
        return (surpass_count / total_count) * 100
    
    def calculate_monthly_surpass_percentage(self, predictions, actual_data, days_ahead):
        """Calculate surpass percentage for monthly predictions"""
        if predictions is None or actual_data is None:
            return None
        
        if len(predictions) < days_ahead:
            return None
        
        surpass_count = 0
        total_count = 0
        
        for i in range(len(predictions) - days_ahead + 1):
            # Get monthly prediction (third target)
            predicted_monthly_close = predictions[i][2]
            
            # Get actual monthly values for the next 'days_ahead' days
            future_data = actual_data.iloc[i:i+days_ahead]
            
            if len(future_data) < days_ahead:
                continue
            
            # Check if predicted monthly close is surpassed
            max_monthly_close = future_data['monthly_close'].max()
            max_monthly_high = future_data['monthly_high'].max()
            max_value = max(max_monthly_close, max_monthly_high)
            
            if max_value > predicted_monthly_close:
                surpass_count += 1
            
            total_count += 1
        
        if total_count == 0:
            return None
        
        return (surpass_count / total_count) * 100
    
    def run_surpass_analysis(self):
        """Run the complete surpass analysis"""
        print("Starting NZDJPY Surpass Analysis (With Pattern Data)")
        print("="*60)
        
        # Load data and model
        self.load_data_and_model()
        self.prepare_scalers()
        
        # Define time horizons
        daily_horizons = [1, 5, 10, 15]
        weekly_horizons = [7, 14, 28, 56]  # 1, 2, 4, 8 weeks
        monthly_horizons = [30, 60, 90]    # 1, 2, 3 months
        
        # Calculate daily surpass percentages
        print("\nCalculating Daily Surpass Percentages...")
        daily_results = {}
        
        for days in daily_horizons:
            # Use a rolling window approach for better statistics
            window_size = 500  # Use last 500 predictions for analysis
            start_idx = max(0, len(self.feature_df) - window_size - self.sequence_length)
            end_idx = len(self.feature_df)
            
            predictions, actual_data = self.make_predictions(start_idx, end_idx)
            
            if predictions is not None and actual_data is not None:
                surpass_pct = self.calculate_surpass_percentage(
                    predictions, actual_data, 'daily_close', 'daily_high', days
                )
                daily_results[f"{days} day(s)"] = surpass_pct
                print(f"  {days} day(s): {surpass_pct:.2f}%" if surpass_pct else f"  {days} day(s): N/A")
            else:
                daily_results[f"{days} day(s)"] = None
                print(f"  {days} day(s): N/A")
        
        # Calculate weekly surpass percentages
        print("\nCalculating Weekly Surpass Percentages...")
        weekly_results = {}
        
        for days in weekly_horizons:
            window_size = 500
            start_idx = max(0, len(self.feature_df) - window_size - self.sequence_length)
            end_idx = len(self.feature_df)
            
            predictions, actual_data = self.make_predictions(start_idx, end_idx)
            
            if predictions is not None and actual_data is not None:
                surpass_pct = self.calculate_weekly_surpass_percentage(
                    predictions, actual_data, days
                )
                weekly_results[f"{days//7} week(s)"] = surpass_pct
                print(f"  {days//7} week(s): {surpass_pct:.2f}%" if surpass_pct else f"  {days//7} week(s): N/A")
            else:
                weekly_results[f"{days//7} week(s)"] = None
                print(f"  {days//7} week(s): N/A")
        
        # Calculate monthly surpass percentages
        print("\nCalculating Monthly Surpass Percentages...")
        monthly_results = {}
        
        for days in monthly_horizons:
            window_size = 500
            start_idx = max(0, len(self.feature_df) - window_size - self.sequence_length)
            end_idx = len(self.feature_df)
            
            predictions, actual_data = self.make_predictions(start_idx, end_idx)
            
            if predictions is not None and actual_data is not None:
                surpass_pct = self.calculate_monthly_surpass_percentage(
                    predictions, actual_data, days
                )
                monthly_results[f"{days//30} month(s)"] = surpass_pct
                print(f"  {days//30} month(s): {surpass_pct:.2f}%" if surpass_pct else f"  {days//30} month(s): N/A")
            else:
                monthly_results[f"{days//30} month(s)"] = None
                print(f"  {days//30} month(s): N/A")
        
        # Create comprehensive results
        self.create_results_summary(daily_results, weekly_results, monthly_results)
        
        return daily_results, weekly_results, monthly_results
    
    def create_results_summary(self, daily_results, weekly_results, monthly_results):
        """Create a comprehensive results summary"""
        print("\n" + "="*80)
        print("NZDJPY LSTM SURPASS PERCENTAGE ANALYSIS (WITH PATTERN DATA)")
        print("="*80)
        print("Percentage of time predicted close is surpassed by actual close or high")
        print("="*80)
        
        # Daily results
        print("\n📈 DAILY PREDICTIONS:")
        print("-" * 40)
        for horizon, pct in daily_results.items():
            if pct is not None:
                print(f"  {horizon:>12}: {pct:6.2f}%")
            else:
                print(f"  {horizon:>12}: N/A")
        
        # Weekly results
        print("\n📊 WEEKLY PREDICTIONS:")
        print("-" * 40)
        for horizon, pct in weekly_results.items():
            if pct is not None:
                print(f"  {horizon:>12}: {pct:6.2f}%")
            else:
                print(f"  {horizon:>12}: N/A")
        
        # Monthly results
        print("\n📅 MONTHLY PREDICTIONS:")
        print("-" * 40)
        for horizon, pct in monthly_results.items():
            if pct is not None:
                print(f"  {horizon:>12}: {pct:6.2f}%")
            else:
                print(f"  {horizon:>12}: N/A")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
        # Create visualization
        self.create_visualization(daily_results, weekly_results, monthly_results)
    
    def create_visualization(self, daily_results, weekly_results, monthly_results):
        """Create visualization of surpass percentages"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Daily results
        daily_horizons = list(daily_results.keys())
        daily_values = [v for v in daily_results.values() if v is not None]
        daily_horizons_clean = [h for h, v in zip(daily_horizons, daily_results.values()) if v is not None]
        
        if daily_values:
            ax1.bar(daily_horizons_clean, daily_values, color='skyblue', alpha=0.7)
            ax1.set_title('Daily Predictions Surpass % (With Patterns)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Time Horizon')
            ax1.set_ylabel('Surpass Percentage (%)')
            ax1.set_ylim(0, 100)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(daily_values):
                ax1.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Weekly results
        weekly_horizons = list(weekly_results.keys())
        weekly_values = [v for v in weekly_results.values() if v is not None]
        weekly_horizons_clean = [h for h, v in zip(weekly_horizons, weekly_results.values()) if v is not None]
        
        if weekly_values:
            ax2.bar(weekly_horizons_clean, weekly_values, color='lightgreen', alpha=0.7)
            ax2.set_title('Weekly Predictions Surpass % (With Patterns)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Time Horizon')
            ax2.set_ylabel('Surpass Percentage (%)')
            ax2.set_ylim(0, 100)
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(weekly_values):
                ax2.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Monthly results
        monthly_horizons = list(monthly_results.keys())
        monthly_values = [v for v in monthly_results.values() if v is not None]
        monthly_horizons_clean = [h for h, v in zip(monthly_horizons, monthly_results.values()) if v is not None]
        
        if monthly_values:
            ax3.bar(monthly_horizons_clean, monthly_values, color='lightcoral', alpha=0.7)
            ax3.set_title('Monthly Predictions Surpass % (With Patterns)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Time Horizon')
            ax3.set_ylabel('Surpass Percentage (%)')
            ax3.set_ylim(0, 100)
            ax3.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(monthly_values):
                ax3.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('nzdjpy_surpass_analysis_with_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to run the surpass analysis"""
    csv_file = 'combined_NZDJPY_daily.csv'
    model_file = 'nzdjpy_lstm_model_with_patterns.h5'
    
    # Create analysis instance
    analyzer = NZDJPYSurpassAnalysisWithPatterns(csv_file, model_file)
    
    # Run analysis
    daily_results, weekly_results, monthly_results = analyzer.run_surpass_analysis()
    
    return analyzer, daily_results, weekly_results, monthly_results

if __name__ == "__main__":
    analyzer, daily_results, weekly_results, monthly_results = main()
