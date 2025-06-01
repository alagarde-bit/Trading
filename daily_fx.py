import requests
import pandas as pd
import os
from datetime import datetime

pairs = [("GBP", "AUD"), ("GBP", "USD"), ("USD", "CHF"), ("USD", "JPY"), ("NZD", "JPY")]
api_key = "R1CNCLWXU1VGFJY4"

for from_symbol, to_symbol in pairs:
    file_name = f"{from_symbol}{to_symbol}_daily.csv"
    
    # Check for existing data
    if os.path.exists(file_name):
        existing_df = pd.read_csv(file_name, index_col=0)  # Assumes timestamp is index
        latest_date = pd.to_datetime(existing_df.index.max())
    else:
        existing_df = pd.DataFrame()
        latest_date = None
    
    # Fetch new data
    url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={from_symbol}&to_symbol={to_symbol}&outputsize=full&apikey={api_key}"
    r = requests.get(url)
    data = r.json()
    time_series = data.get("Time Series FX (Daily)", {})
    
    # Convert to DataFrame
    new_df = pd.DataFrame.from_dict(time_series, orient="index")
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
        # Sort by date in ascending order
        combined_df = combined_df.sort_index(ascending=True)
        
        # Write the sorted DataFrame to CSV (overwrite mode)
        combined_df.to_csv(file_name)
        print(f"Updated {from_symbol}/{to_symbol} daily with {len(new_df)} new entries")
    else:
        print(f"No new data for {from_symbol}/{to_symbol} daily")