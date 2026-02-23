import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class NZDJPYLSTMPredictorEnhancedV2:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.model = None
        self.scaler = MinMaxScaler()  # For targets (diffs)
        self.feature_scaler = MinMaxScaler()  # For features
        self.pattern_mlbs = {}  # Dictionary for MultiLabelBinarizer per column
        self.sequence_length = 30  # Reduced for speed
        self.prediction_cutoff = '2024-01-01'  # Temporal split for train/test
        self.numerical_columns = []
        self.categorical_columns = []
        self.encoded_feature_columns = []  # Track after encoding
        self.numerical_columns_base = []  # Base numerical before encoding
        
    def load_and_preprocess_data(self):
        """Load and preprocess the NZDJPY data"""
        print("Loading data...")
        self.df = pd.read_csv(self.csv_file)
        
        # Convert date column to datetime
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Sort by date
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        print(f"Data loaded: {len(self.df)} records from {self.df['date'].min()} to {self.df['date'].max()}")
        
        # Define current pattern columns only (remove prev_* for redundancy)
        pattern_columns = [
            'daily_candlestick_patterns', 'daily_chart_patterns',
            'weekly_candlestick_patterns', 'weekly_chart_patterns',
            'monthly_candlestick_patterns', 'monthly_chart_patterns'
        ]
        
        # Check which have data
        available_pattern_columns = []
        for col in pattern_columns:
            if col in self.df.columns:
                non_null_count = self.df[col].notna().sum()
                if non_null_count > 100:
                    available_pattern_columns.append(col)
                    print(f"  {col}: {non_null_count} non-null values")
        
        print(f"Using {len(available_pattern_columns)} pattern columns with sufficient data")
        
        # Numerical features
        numerical_features = [
            'daily_open', 'daily_high', 'daily_low', 'daily_close',
            'weekly_open', 'weekly_high', 'weekly_low', 'weekly_close',
            'monthly_open', 'monthly_high', 'monthly_low', 'monthly_close',
            'daily_sma_20', 'daily_sma_50', 'daily_sma_200', 'daily_rsi',
            'weekly_sma_4', 'weekly_sma_10', 'weekly_sma_40', 'weekly_rsi',
            'monthly_sma_1', 'monthly_sma_3', 'monthly_sma_10', 'monthly_rsi'
        ]
        
        self.numerical_columns_base = [col for col in numerical_features if col in self.df.columns]
        self.categorical_columns = available_pattern_columns
        
        # All feature columns (pre-encoding)
        feature_columns = self.numerical_columns_base + self.categorical_columns
        
        # Filter out NaN values more selectively - only require basic price data
        required_columns = ['date', 'daily_open', 'daily_high', 'daily_low', 'daily_close', 
                           'weekly_open', 'weekly_high', 'weekly_low', 'weekly_close',
                           'monthly_open', 'monthly_high', 'monthly_low', 'monthly_close']
        
        # First, ensure we have all basic price data
        self.feature_df = self.df[required_columns].dropna()
        print(f"After removing NaN from basic price data: {len(self.feature_df)} records")
        
        # Then add other features, filling NaN with 0 for technical indicators
        for col in feature_columns:
            if col not in required_columns and col in self.df.columns:
                self.feature_df[col] = self.df[col]
                # Fill NaN technical indicators with 0 (neutral values)
                if col in self.numerical_columns_base:
                    self.feature_df[col] = self.feature_df[col].fillna(0)
        
        print(f"After adding technical indicators (filled NaN with 0): {len(self.feature_df)} records")
        
        # Split data at cutoff date
        cutoff_date = pd.to_datetime(self.prediction_cutoff)
        self.train_data = self.feature_df[self.feature_df['date'] <= cutoff_date].copy()
        self.test_data = self.feature_df[self.feature_df['date'] > cutoff_date].copy()
        
        print(f"Training data: {len(self.train_data)} records (up to {self.prediction_cutoff})")
        print(f"Test data: {len(self.test_data)} records (after {self.prediction_cutoff})")
        
        return self.train_data, self.test_data
    
    def encode_patterns(self, data):
        """Encode pattern columns using MultiLabelBinarizer"""
        data_encoded = data.copy()
        encoded_cols = []  # Local list for this call
        
        for col in self.categorical_columns:
            if col in data_encoded.columns:
                # Fill NaN/empty with empty list for no patterns
                data_encoded[col] = data_encoded[col].fillna('')
                # Split into lists
                pattern_lists = data_encoded[col].str.split(';').apply(lambda x: [p.strip() for p in x if p.strip()])
                
                # Create or use existing ML Binarizer (fit on train)
                if col not in self.pattern_mlbs:
                    # Fit on train data's patterns
                    train_lists = self.train_data[col].fillna('').str.split(';').apply(lambda x: [p.strip() for p in x if p.strip()])
                    self.pattern_mlbs[col] = MultiLabelBinarizer()
                    self.pattern_mlbs[col].fit(train_lists)
                
                # Transform
                encoded = self.pattern_mlbs[col].transform(pattern_lists)
                encoded_df = pd.DataFrame(encoded, 
                                        columns=[f"{col}_{cls}" for cls in self.pattern_mlbs[col].classes_], 
                                        index=data_encoded.index)
                
                # Concat and drop original
                data_encoded = pd.concat([data_encoded.drop(col, axis=1), encoded_df], axis=1)
                encoded_cols.extend(encoded_df.columns.tolist())
        
        # For this encoded data, numerical_columns = base + encoded
        self.numerical_columns = self.numerical_columns_base + encoded_cols
        
        return data_encoded
    
    def create_sequences(self, data, target_columns, fit_scalers=True):
        """Create sequences for LSTM training with differenced targets"""
        # Encode patterns (now includes binary features)
        data_encoded = self.encode_patterns(data)
        
        # Features (exclude date and targets; now numerical includes encoded binaries)
        feature_columns = [col for col in data_encoded.columns if col not in ['date'] + target_columns]
        features = data_encoded[feature_columns].values
        
        # Targets: compute differences for stationarity
        targets_df = data_encoded[target_columns]
        targets_diff = targets_df.diff().fillna(0).values  # First diff is 0, rest are changes
        
        # Scale features
        if fit_scalers:
            features_scaled = self.feature_scaler.fit_transform(features)
        else:
            features_scaled = self.feature_scaler.transform(features)
        
        # Scale targets (diffs)
        if fit_scalers:
            targets_scaled = self.scaler.fit_transform(targets_diff)
        else:
            targets_scaled = self.scaler.transform(targets_diff)
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(features_scaled)):
            X.append(features_scaled[i-self.sequence_length:i])
            y.append(targets_scaled[i])
        
        return np.array(X), np.array(y), targets_df.values  # Return original absolutes for reconstruction
    
    def build_model(self, input_shape, output_dim):
        """Build LSTM model with L2 regularization"""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(64, return_sequences=True, dropout=0.3, kernel_regularizer=l2(0.01)),
            LSTM(32, dropout=0.3, kernel_regularizer=l2(0.01)),
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(output_dim, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self):
        """Train the LSTM model"""
        print("Preparing training data...")
        
        target_columns = ['daily_close', 'weekly_close', 'monthly_close']
        
        X_train, y_train, _ = self.create_sequences(self.train_data, target_columns, fit_scalers=True)
        
        print(f"Training sequences shape: {X_train.shape}")
        print(f"Training targets shape: {y_train.shape}")
        
        if len(X_train) == 0:
            raise ValueError("No training sequences available. Check data quality.")
        
        self.model = self.build_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            output_dim=y_train.shape[1]
        )
        
        print("Model architecture:")
        self.model.summary()
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.0001
        )
        
        print("Training model...")
        history = self.model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=50,  # Reduced for speed
            batch_size=64,  # Increased for speed
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Plot training history
        self.plot_training_history(history)
        
        return history
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(history.history['mae'], label='Training MAE')
        ax2.plot(history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('nzdjpy_training_history_enhanced_v2.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def make_predictions(self):
        """Make predictions for the next values (reconstruct from diffs)"""
        print("Making predictions...")
        
        target_columns = ['daily_close', 'weekly_close', 'monthly_close']
        
        last_sequence_data = self.train_data.tail(self.sequence_length).copy()
        last_sequence_encoded = self.encode_patterns(last_sequence_data)
        
        feature_columns = [col for col in last_sequence_encoded.columns if col not in ['date'] + target_columns]
        last_sequence = last_sequence_encoded[feature_columns].values
        last_sequence_scaled = self.feature_scaler.transform(last_sequence)
        
        X_pred = last_sequence_scaled.reshape(1, self.sequence_length, -1)
        
        prediction_diff_scaled = self.model.predict(X_pred, verbose=0)
        prediction_diff = self.scaler.inverse_transform(prediction_diff_scaled)
        
        last_values = self.train_data[target_columns].iloc[-1].values
        prediction = last_values + prediction_diff[0]  # Add diffs to last absolutes
        
        print("\n" + "="*60)
        print("NZDJPY PREDICTIONS (Enhanced with Patterns V2)")
        print("="*60)
        print(f"Last known values ({self.prediction_cutoff}):")
        for i, col in enumerate(target_columns):
            print(f"  {col}: {last_values[i]:.5f}")
        
        print(f"\nPredicted next values:")
        for i, col in enumerate(target_columns):
            change = prediction[i] - last_values[i]
            change_pct = (change / last_values[i]) * 100
            print(f"  {col}: {prediction[i]:.5f} ({change:+.5f}, {change_pct:+.2f}%)")
        
        print("="*60)
        
        return prediction, last_values
    
    def evaluate_model(self):
        """Evaluate model performance on test data if available (reconstruct from diffs)"""
        if len(self.test_data) == 0:
            print("No test data available for evaluation.")
            return
        
        print("Evaluating model on test data...")
        
        target_columns = ['daily_close', 'weekly_close', 'monthly_close']
        
        X_test, y_test_diff_scaled, y_test_actual = self.create_sequences(self.test_data, target_columns, fit_scalers=False)
        
        if len(X_test) == 0:
            print("Not enough test data for evaluation.")
            return
        
        y_pred_diff_scaled = self.model.predict(X_test, verbose=0)
        y_pred_diff = self.scaler.inverse_transform(y_pred_diff_scaled)
        
        # Reconstruct predicted absolutes: start from initial test value + cumsum diffs
        initial_values = y_test_actual[self.sequence_length-1:self.sequence_length]
        y_pred_actual = np.cumsum(y_pred_diff, axis=0) + initial_values
        
        mse = mean_squared_error(y_test_actual[self.sequence_length:], y_pred_actual)
        mae = mean_absolute_error(y_test_actual[self.sequence_length:], y_pred_actual)
        rmse = np.sqrt(mse)
        
        print(f"\nModel Performance on Test Data:")
        print(f"RMSE: {rmse:.5f}")
        print(f"MAE: {mae:.5f}")  # Optimized metric
        
        self.plot_predictions(y_test_actual[self.sequence_length:], y_pred_actual, target_columns)
    
    def plot_predictions(self, y_actual, y_pred, target_columns):
        """Plot predictions vs actual values"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, col in enumerate(target_columns):
            axes[i].plot(y_actual[:, i], label='Actual', alpha=0.7)
            axes[i].plot(y_pred[:, i], label='Predicted', alpha=0.7)
            axes[i].set_title(f'{col} - Actual vs Predicted')
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel('Price')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig('nzdjpy_predictions_evaluation_enhanced_v2.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filename='nzdjpy_lstm_model_enhanced_v2'):
        """Save the trained model"""
        self.model.save(f'{filename}.h5')
        print(f"Model saved as {filename}.h5")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting NZDJPY LSTM Prediction Analysis (Enhanced with Patterns V2)")
        print("="*70)
        
        self.load_and_preprocess_data()
        
        history = self.train_model()
        
        predictions, last_values = self.make_predictions()
        
        self.evaluate_model()
        
        self.save_model()
        
        print("\nAnalysis complete!")
        return predictions, last_values

def main():
    """Main function to run the analysis"""
    csv_file = 'combined_NZDJPY_daily.csv'
    
    predictor = NZDJPYLSTMPredictorEnhancedV2(csv_file)
    
    predictions, last_values = predictor.run_complete_analysis()
    
    return predictor, predictions, last_values

if __name__ == "__main__":
    predictor, predictions, last_values = main()