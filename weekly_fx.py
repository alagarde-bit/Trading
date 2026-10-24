import requests
import pandas as pd
import os
import time
from datetime import datetime

pairs = [("GBP", "AUD"), ("GBP", "USD"), ("USD", "CHF"), ("USD", "JPY"), ("NZD", "JPY")]
api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
if not api_key:
    raise SystemExit("Set ALPHA_VANTAGE_API_KEY (see .env.example) before fetching FX data.")

for from_symbol, to_symbol in pairs:
    file_name = f"{from_symbol}{to_symbol}_weekly.csv"
    
    # Check for existing data
    if os.path.exists(file_name):
        existing_df = pd.read_csv(file_name, index_col=0)  # Assumes timestamp is index
        latest_date = pd.to_datetime(existing_df.index.max())
    else:
        existing_df = pd.DataFrame()
        latest_date = None
    
    # Fetch new data
    url = f"https://www.alphavantage.co/query?function=FX_WEEKLY&from_symbol={from_symbol}&to_symbol={to_symbol}&apikey={api_key}"
    r = requests.get(url)
    data = r.json()
    time_series = data.get("Time Series FX (Weekly)", {})
    
    # Convert to DataFrame
    new_df = pd.DataFrame.from_dict(time_series, orient="index")
    if new_df.empty:
        print(f"No new data for {from_symbol}/{to_symbol} weekly")
        time.sleep(1.5)  # Rate limit: wait before next request
        continue
    if len(new_df.columns) != 4:
        print(f"Unexpected weekly format for {from_symbol}/{to_symbol} (got {len(new_df.columns)} columns)")
        time.sleep(1.5)
        continue
    new_df.columns = ["open", "high", "low", "close"]
    new_df.index = pd.to_datetime(new_df.index)
    
    # Exclude current date
    current_date = pd.to_datetime(datetime.now().date())
    new_df = new_df[new_df.index < current_date]
    
    # Filter out duplicates based on latest date
    if latest_date:
        new_df = new_df[new_df.index > latest_date]
    
    # Append new data and ensure ascending order
    if not new_df.empty:
        # Combine existing and new data
        combined_df = pd.concat([existing_df, new_df]) if not existing_df.empty else new_df
        combined_df.index = pd.to_datetime(combined_df.index)
        # Sort by date in ascending order
        combined_df = combined_df.sort_index(ascending=True)
        
        # Write the sorted DataFrame to CSV (overwrite mode)
        combined_df.to_csv(file_name)
        print(f"Updated {from_symbol}/{to_symbol} weekly with {len(new_df)} new entries")
    else:
        print(f"No new data for {from_symbol}/{to_symbol} weekly")
    time.sleep(1.5)  # Rate limit: wait before next request