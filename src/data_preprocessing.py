from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import pandas as pd


class DataPreprocessor:
    """Class responsible for all data preprocessing tasks."""
    
    def __init__(self, acute_window='7D', chronic_window='28D'):
        """Initialize preprocessor with configurable rolling windows."""
        self.acute_window = acute_window
        self.chronic_window = chronic_window
        self.session_types = ['match', 'training']
        
    def _identify_session_type(self, tags):
        """Extract session type from Tags field."""
        if isinstance(tags, str):
            match_patterns = ['match', 'game', 'fixture', 'competition']
            for pattern in match_patterns:
                if pattern in tags.lower():
                    return 'match'
        return 'training'
        
    def _process_player_data(self, player_data):
        """Process data for a single player."""
        if len(player_data) < 5:  # Skip players with insufficient data
            return None
            
        # Reset index to avoid any issues with duplicate indices
        player_data = player_data.reset_index(drop=True)
            
        # Sort data by date
        player_data = player_data.sort_values('Date')
        
        # Extract session type from Tags
        player_data['Session_Type'] = player_data['Tags'].apply(self._identify_session_type)
        
        # Calculate separate load metrics for training sessions
        training_data = player_data[player_data['Session_Type'] == 'training']
        
        if not training_data.empty:
            # Handle possible duplicate dates by creating a new unique column
            # Combine date and a sequential counter to ensure uniqueness
            temp_training = training_data.copy().reset_index(drop=True)
            temp_training['Date_Unique'] = [f"{d}_{i}" for i, d in enumerate(temp_training['Date'])]
            temp_training_indexed = temp_training.set_index('Date')
            
            # Calculate rolling values without reindexing
            acute_load = temp_training_indexed['Player Load'].rolling(window=self.acute_window, min_periods=1).sum()
            chronic_load = temp_training_indexed['Player Load'].rolling(window=self.chronic_window, min_periods=1).sum()
            
            # Map values back to original dataframe using the row positions
            player_data['Training_Acute_Load'] = np.nan
            player_data['Training_Chronic_Load'] = np.nan
            
            # Assign calculated values back to training rows
            for i, idx in enumerate(training_data.index):
                if i < len(acute_load):
                    player_data.loc[idx, 'Training_Acute_Load'] = acute_load.iloc[i]
                    player_data.loc[idx, 'Training_Chronic_Load'] = chronic_load.iloc[i]
            
            # Forward fill for non-training sessions
            player_data['Training_Acute_Load'] = player_data['Training_Acute_Load'].fillna(method='ffill')
            player_data['Training_Chronic_Load'] = player_data['Training_Chronic_Load'].fillna(method='ffill')
            
            # Calculate ACWR for training
            player_data['Training_ACWR'] = player_data['Training_Acute_Load'] / player_data['Training_Chronic_Load'].replace(0, np.nan)
            player_data['Training_ACWR'] = player_data['Training_ACWR'].fillna(1)
        
        # Calculate overall load metrics (all sessions)
        # Create temporary series for calculations that doesn't depend on Date as index
        temp_series = player_data['Player Load'].copy().reset_index(drop=True)
        
        # Calculate rolling values directly on the Series
        acute_load = temp_series.rolling(window=7, min_periods=1).sum()  # Use 7 days for acute
        chronic_load = temp_series.rolling(window=28, min_periods=1).sum()  # Use 28 days for chronic
        
        # Assign calculated values directly
        player_data['Acute_Load'] = acute_load.values
        player_data['Chronic_Load'] = chronic_load.values
        
        # Calculate ACWR for all sessions
        player_data['ACWR'] = player_data['Acute_Load'] / player_data['Chronic_Load'].replace(0, np.nan)
        player_data['ACWR'] = player_data['ACWR'].fillna(1)
        
        # Calculate load monotony and strain using non-indexed data
        temp_series = player_data['Player Load'].copy().reset_index(drop=True)
        std_series = temp_series.rolling(window=7, min_periods=1).std()
        mean_series = temp_series.rolling(window=7, min_periods=1).mean()
        
        # Calculate monotony safely avoiding division by zero
        load_monotony = std_series / mean_series
        player_data['Load_Monotony'] = load_monotony.fillna(0).replace([np.inf, -np.inf], 0).values
        player_data['Load_Strain'] = player_data['Player Load'] * player_data['Load_Monotony']
        
        # Create high-intensity action metrics
        player_data['High_Speed_Distance'] = player_data['Distance in Speed Zone 4  (metres)'] + player_data['Distance in Speed Zone 5  (metres)']
        player_data['High_Accel_Count'] = player_data['Accelerations Zone Count: 3 - 4 m/s/s'] + player_data['Accelerations Zone Count: > 4 m/s/s']
        player_data['High_Decel_Count'] = player_data['Deceleration Zone Count: 3 - 4 m/s/s'] + player_data['Deceleration Zone Count: > 4 m/s/s']
        player_data['High_Impact_Count'] = player_data['Impact Zones: 10 - 15 G (Impacts)'] + player_data['Impact Zones: 15 - 20 G (Impacts)'] + player_data['Impact Zones: > 20 G (Impacts)']
        
        # Calculate power metrics if columns exist
        high_power_columns = [f'Distance in Power Zone: {i} - {i+5} w/kg  (metres)' for i in range(25, 50, 5)]
        if all(col in player_data.columns for col in high_power_columns):
            player_data['High_Power_Distance'] = player_data[high_power_columns].sum(axis=1) + player_data['Distance in Power Zone: > 50 w/kg  (metres)']
        
        # Calculate fatigue indicators for each session type
        for session_type in self.session_types:
            type_data = player_data[player_data['Session_Type'] == session_type]
            if len(type_data) > 1:
                # Speed decline within session type
                player_data[f'{session_type.capitalize()}_Speed_Decline'] = np.nan
                
                # Calculate using vectorized operations where possible
                for idx, row in type_data.iterrows():
                    # Get last 7 days of same session type
                    last_7_days = type_data[(type_data['Date'] <= row['Date']) & 
                                            (type_data['Date'] >= row['Date'] - pd.Timedelta(days=7))]
                    
                    if len(last_7_days) > 1:
                        max_speed = last_7_days['Top Speed (m/s)'].max()
                        current_speed = row['Top Speed (m/s)']
                        decline = (current_speed - max_speed) / max_speed if max_speed > 0 else 0
                        player_data.loc[idx, f'{session_type.capitalize()}_Speed_Decline'] = decline
        
        # Calculate performance metrics relative to individual baselines
        for session_type in self.session_types:
            type_data = player_data[player_data['Session_Type'] == session_type]
            if len(type_data) > 4:  # Need enough data to establish baseline
                for metric in ['Distance Per Min (m/min)', 'Sprints Per Min', 'Power Score (w/kg)']:
                    if metric in type_data.columns:
                        baseline = type_data[metric].quantile(0.75)  # 75th percentile as "good" performance
                        column_name = f'{session_type.capitalize()}_{metric}_vs_Baseline'
                        player_data[column_name] = np.nan
                        player_data.loc[type_data.index, column_name] = type_data[metric] / baseline if baseline > 0 else np.nan
        
        # Add days since last match
        player_data['Days_Since_Match'] = np.nan
        match_dates = player_data[player_data['Session_Type'] == 'match']['Date'].sort_values()
        
        if len(match_dates) > 0:
            for idx, row in player_data.iterrows():
                # Find most recent match before this session
                prev_matches = match_dates[match_dates < row['Date']]
                if len(prev_matches) > 0:
                    most_recent_match = prev_matches.max()
                    days_since = (row['Date'] - most_recent_match).days
                    player_data.loc[idx, 'Days_Since_Match'] = days_since
        
        # Add days until next match
        player_data['Days_Until_Match'] = np.nan
        
        for idx, row in player_data.iterrows():
            # Find next match after this session
            next_matches = match_dates[match_dates > row['Date']]
            if len(next_matches) > 0:
                next_match = next_matches.min()
                days_until = (next_match - row['Date']).days
                player_data.loc[idx, 'Days_Until_Match'] = days_until
        
        return player_data
        
    def process_data(self, data):
        """Process the entire dataset."""
        # Make a copy to avoid modifying original data
        data = data.copy()
        
        # Convert date format
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Process each player's data separately, potentially in parallel
        players = data['Player Name'].unique()
        processed_data = []
        
        # For smaller datasets, use direct processing
        if len(players) <= 10:
            for player in players:
                player_data = data[data['Player Name'] == player]
                processed_player_data = self._process_player_data(player_data)
                if processed_player_data is not None:
                    # Ensure the DataFrame has no index issues before appending
                    processed_player_data = processed_player_data.reset_index(drop=True)
                    processed_data.append(processed_player_data)
        else:
            # For larger datasets, use parallel processing
            num_cores = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
            
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                futures = []
                for player in players:
                    player_data = data[data['Player Name'] == player]
                    futures.append(executor.submit(self._process_player_data, player_data))
                
                for future in futures:
                    result = future.result()
                    if result is not None:
                        # Ensure the DataFrame has no index issues before appending
                        result = result.reset_index(drop=True)
                        processed_data.append(result)
        
        if processed_data:
            # Make sure all DataFrames have proper indices before concatenation
            safe_concat_data = []
            for df in processed_data:
                # Check for duplicate indices and reset if found
                if df.index.duplicated().any():
                    df = df.reset_index(drop=True)
                safe_concat_data.append(df)
            
            # Use ignore_index=True to completely avoid index-related issues
            return pd.concat(safe_concat_data, ignore_index=True)
        else:
            return pd.DataFrame()  # Return empty DataFrame if no players had sufficient data