from datetime import datetime, timedelta
import json
import pandas as pd
from config import TEMP_PROCESSED_JSON


def data_preprocessing(path):
    """
    This function reads from news.csv and extracts only the subset of news data from yesterday to today.
    After extracting, df_subset is saved in temp_processed_data.json.
    
    Args:
        path(str): Path to news.csv
    """
    try:
        # Load the CSV file
        df = pd.read_csv(path, delimiter=";", encoding='utf-8')
        
        # Rename columns accordingly
        df = df.rename(columns={
            "Date and Timestamp": "datetime",
            "Title": "header",
            "Full Text": "content",
            "Source": "source",
            "Link": "link",
        })

        # Reformat the date column to datetime object and sort in ascending order
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by='datetime', ascending=True)
        df = df.reset_index(drop=True)

        # Set startdate as yesterday's date and end date as today's date
        start_date = (datetime.now() - timedelta(days=1)).date()
        end_date = datetime.now().date()
        print(f'Start date: {start_date}')
        print(f'End date: {end_date}')
        
        df_subset = df[(df['datetime'].dt.date >= start_date) & (df['datetime'].dt.date < end_date)]
        df_subset.reset_index(drop=True)
        
        # Check if df_subset is empty
        if df_subset.empty:
            raise ValueError("No data available for the given date range. df_subset is empty.")

        # Convert each row to a dictionary and collect in a list
        data_dicts = df_subset.to_dict(orient='records')

        # Save to JSON file
        with open(TEMP_PROCESSED_JSON, 'w') as f:
            json.dump(data_dicts, f, indent=4, default=str)
            print('Processed data saved as JSON file')

    except Exception as e:
        print(f"[Error in data preprocessing pipeline] -> {e}")
  
      