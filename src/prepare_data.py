import pandas as pd
import numpy as np

def load_and_preprocess_data(data_path):
    """
    Loads data from a CSV file, performs initial preprocessing, and creates a target variable
    based on performance fluctuations.
    """
    df = pd.read_csv(data_path)
    
    # Convert Date to datetime objects
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort data by Player Name and Date
    df = df.sort_values(by=['Player Name', 'Date'])

    # Calculate a 5-session rolling average of 'Distance Per Min (m/min)' for each player
    df['Rolling_Avg_Distance_Per_Min'] = df.groupby('Player Name')['Distance Per Min (m/min)'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)

    # Create the binary 'target' column
    # Target is 1 if current performance is less than 90% of the rolling average
    # Handle initial sessions where rolling average might be NaN or 0
    df['target'] = np.where((df['Distance Per Min (m/min)'] < 0.9 * df['Rolling_Avg_Distance_Per_Min']) & (df['Rolling_Avg_Distance_Per_Min'] > 0), 1, 0)
    
    # Drop the intermediate rolling average column
    df = df.drop(columns=['Rolling_Avg_Distance_Per_Min'])
    return df

def load_and_prepare_data(filepath):
    """Load and prepare the raw data for analysis."""
    data = pd.read_csv(filepath)

    # Check for required columns
    required_columns = ['Date', 'Player Name', 'Tags', 'Player Load', 'Distance Per Min (m/min)']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Add Position column if it doesn't exist (algorithm uses it, but it's optional)
    if 'Position' not in data.columns:
        data['Position'] = 'Unknown'
    
    return data

if __name__ == "__main__":
    # Run this script to test data loading
    data = load_and_prepare_data("data/raw/report_export.csv")
    print(f"Successfully loaded data with {len(data)} rows and {len(data.columns)} columns")
    print(f"Players in dataset: {data['Player Name'].nunique()}")