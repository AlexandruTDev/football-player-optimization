import pandas as pd

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