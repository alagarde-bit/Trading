import requests
import pandas as pd
import os

pairs = [("GBP", "AUD"), ("GBP", "USD"), ("USD", "CHF"), ("USD", "JPY"), ("NZD", "JPY")]
api_key = "your_api_key_here"

for from_symbol, to_symbol in pairs:
    file_name = f"{from_symbol}{to_symbol}_weekly.csv"
    
    # Check for existing data
    if os.path.exists(file_name):
        existing_df = pd.read_csv(file_name, index_col=0)
        latest_date = pd.to_datetime(existing_df.index.max())
    else:
        latest_date = None
    
    # Fetch new data
    url = f"https://www.alphavantage.co/query?function=FX_WEEKLY&from_symbol={from_symbol}&to_symbol={to_symbol}&apikey={api_key}"
    r = requests.get(url)
    data = r.json()
    time_series = data.get("Time Series FX (Weekly)", {})
    
    # Convert to DataFrame
    new_df = pd.DataFrame.from_dict(time_series, orient="index")
    new_df.columns = ["open", "high", "low", "close"]
    new_df.index = pd.to_datetime(new_df.index)
    
    # Filter out duplicates
    if latest_date:
        new_df = new_df[new_df.index > latest_date]
    
    # Append new data to existing file
    if not new_df.empty:
        if os.path.exists(file_name):
            new_df.to_csv(file_name, mode='a', header=False)
        else:
            new_df.to_csv(file_name)
        print(f"Updated {from_symbol}/{to_symbol} weekly with {len(new_df)} new entries")
    else:
        print(f"No new data for {from_symbol}/{to_symbol} weekly")