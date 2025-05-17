from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split


class ModelTrainer:
    """Class responsible for training player performance models."""
    
    def __init__(self):
        """Initialize model training components."""
        self.load_models = {}  # Player-specific models for optimal load
        self.fatigue_models = {'match': {}, 'training': {}}  # Separate models for match/training
        self.taper_models = {}  # Player-specific models for tapering strategies
        self.scaler = StandardScaler()
        self.session_types = ['match', 'training']
        
    def identify_optimal_load_zones(self, data, performance_metrics=['Distance Per Min (m/min)', 'Sprints Per Min', 'Power Score (w/kg)']):
        """Identify optimal training load zones for peak performance."""
        players = data['Player Name'].unique()
        optimal_zones = {}
        
        # Process each player individually
        for player in players:
            player_data = data[data['Player Name'] == player]
            
            # Separate match and training data
            match_data = player_data[player_data['Session_Type'] == 'match']
            training_data = player_data[player_data['Session_Type'] == 'training']
            
            if len(match_data) < 5:  # Need enough matches
                continue
                
            # For each performance metric
            player_zones = {
                'match': {},
                'training': {}
            }
            
            for metric in performance_metrics:
                if metric not in match_data.columns:
                    continue
                    
                # Define "good match performance" as top 25% of match performances
                threshold = match_data[metric].quantile(0.75) if len(match_data) >= 4 else None
                if threshold is not None:
                    good_matches = match_data[match_data[metric] >= threshold]
                    
                    if len(good_matches) < 3:  # Not enough data
                        continue
                        
                    # Look at training loads in week leading up to good matches
                    good_dates = good_matches['Date'].tolist()
                    relevant_sessions = []
                    
                    for date in good_dates:
                        # Get training sessions 1-7 days before good match
                        prior_sessions = training_data[(training_data['Date'] >= date - timedelta(days=7)) & 
                                                    (training_data['Date'] < date)]
                        if not prior_sessions.empty:
                            relevant_sessions.append(prior_sessions)
                    
                    if not relevant_sessions:
                        continue
                        
                    # Analyze common load patterns in these sessions
                    optimal_sessions = pd.concat(relevant_sessions)
                    
                    # Group by days before match
                    optimal_sessions['Days_Before_Match'] = optimal_sessions['Days_Until_Match']
                    
                    # Extract optimal zones
                    for days_before in range(1, 8):
                        day_sessions = optimal_sessions[optimal_sessions['Days_Before_Match'] == days_before]
                        if len(day_sessions) >= 3:
                            player_zones['training'][f'MD-{days_before}'] = {
                                'daily_load': (day_sessions['Player Load'].quantile(0.25), 
                                              day_sessions['Player Load'].quantile(0.75)),
                                'acwr': (day_sessions['Training_ACWR'].quantile(0.25), 
                                        day_sessions['Training_ACWR'].quantile(0.75)),
                                'high_speed_dist': (day_sessions['High_Speed_Distance'].quantile(0.25), 
                                                  day_sessions['High_Speed_Distance'].quantile(0.75)),
                                'high_intensity_actions': (day_sessions['High_Accel_Count'].quantile(0.25), 
                                                         day_sessions['High_Accel_Count'].quantile(0.75))
                            }
                    
                    # Add match day optimal zones
                    player_zones['match'][metric] = {
                        'distance_per_min': (good_matches['Distance Per Min (m/min)'].quantile(0.25),
                                            good_matches['Distance Per Min (m/min)'].quantile(0.75)),
                        'sprint_count': (good_matches['Sprints'].quantile(0.25),
                                        good_matches['Sprints'].quantile(0.75)),
                        'high_intensity_distance': (good_matches['High_Speed_Distance'].quantile(0.25),
                                                  good_matches['High_Speed_Distance'].quantile(0.75))
                    }
            
            if player_zones['training'] or player_zones['match']:
                optimal_zones[player] = player_zones
                
        return optimal_zones

    def _train_player_fatigue_model(self, player, player_data, session_type):
        """Train fatigue model for a specific player and session type."""
        type_data = player_data[player_data['Session_Type'] == session_type].sort_values('Date')
        
        if len(type_data) < 20:  # Need sufficient data
            return None
            
        # Define features for fatigue prediction
        features = ['ACWR', 'Load_Monotony', 'Load_Strain', 'High_Speed_Distance', 
                    'High_Accel_Count', 'High_Decel_Count', 'High_Impact_Count',
                    'Days_Since_Match']
        
        if session_type == 'training':
            # Add training-specific features
            features.extend(['Training_ACWR', 'Days_Until_Match'])
        
        # Keep only features that exist in the data
        features = [f for f in features if f in type_data.columns]
        
        if not features:
            return None
            
        # Target: Speed decline in next session of same type
        target_col = f'{session_type.capitalize()}_Speed_Decline'
        if target_col not in type_data.columns:
            return None
            
        type_data[f'Next_{target_col}'] = type_data[target_col].shift(-1)
        type_data['Performance_Drop'] = (type_data[f'Next_{target_col}'] < -0.05).astype(int)
        
        # Prepare training data
        X = type_data[features].dropna()
        y = type_data['Performance_Drop'].loc[X.index].dropna()
        
        # Make sure X and y have same indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(X) < 15:  # Not enough samples after cleaning
            return None
            
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1]
        }
        
        # Train fatigue prediction model with hyperparameter tuning
        model = GradientBoostingClassifier(random_state=42)
        
        # Use cross-validation if enough data
        if len(X) >= 30:
            # Create temporal splits to preserve time-series nature
            cv = GroupKFold(n_splits=min(5, len(X) // 6))
            time_groups = np.arange(len(X)) // 6  # Approximating time periods
            
            grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy')
            grid_search.fit(X, y, groups=time_groups)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_accuracy = grid_search.best_score_
            
            # Final evaluation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Get feature importances
            result = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
            feature_importance = {features[i]: result.importances_mean[i] for i in range(len(features))}
            
            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            return {
                'model': best_model,
                'accuracy': accuracy,
                'cv_accuracy': cv_accuracy,
                'features': features,
                'best_params': best_params,
                'feature_importance': feature_importance,
                'confusion_matrix': cm
            }
        else:
            # Simple train/test split for smaller datasets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            if accuracy > 0.6:  # Only keep reasonably accurate models
                # Get feature importances
                result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
                feature_importance = {features[i]: result.importances_mean[i] for i in range(len(features))}
                                
                return {
                    'model': model,
                    'accuracy': accuracy,
                    'features': features,
                    'feature_importance': feature_importance,
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }
            
        return None
                
    def train_fatigue_prediction_model(self, data):
        """Train models to predict performance declines based on accumulated fatigue."""
        players = data['Player Name'].unique()
        
        # Train separate models for match and training fatigue
        for session_type in self.session_types:
            self.fatigue_models[session_type] = {}
            
            # Train a separate model for each player and session type
            for player in players:
                player_data = data[data['Player Name'] == player]
                model_info = self._train_player_fatigue_model(player, player_data, session_type)
                
                if model_info is not None:
                    self.fatigue_models[session_type][player] = model_info
    
    def develop_tapering_strategies(self, data):
        """Develop personalized tapering strategies before matches."""
        players = data['Player Name'].unique()
        taper_strategies = {}
        
        # Process each player individually
        for player in players:
            player_data = data[data['Player Name'] == player].copy()
            
            # Skip players with insufficient data
            if len(player_data) < 15:
                continue
                
            # Identify match days
            match_data = player_data[player_data['Session_Type'] == 'match']
            
            if len(match_data) < 5:  # Need enough matches
                continue
            
            # Define "good" match performances (top 33%)
            metrics_to_check = ['Distance Per Min (m/min)', 'Sprints Per Min']
            metrics_in_data = [m for m in metrics_to_check if m in match_data.columns]
            
            if not metrics_in_data:
                continue
                
            good_matches = match_data.copy()
            
            for metric in metrics_in_data:
                threshold = match_data[metric].quantile(0.67)
                good_matches = good_matches[good_matches[metric] >= threshold]
            
            if len(good_matches) < 3:
                continue
                
            # Find all training sessions in 7 days before these good matches
            pre_match_sessions = []
            
            for _, match in good_matches.iterrows():
                match_date = match['Date']
                pre_match = player_data[(player_data['Date'] < match_date) & 
                                       (player_data['Date'] >= match_date - timedelta(days=7)) &
                                       (player_data['Session_Type'] == 'training')]
                
                for _, session in pre_match.iterrows():
                    days_before = (match_date - session['Date']).days
                    session_dict = {
                        'days_before_match': days_before,
                        'player_load': session['Player Load'],
                        'high_speed_distance': session['High_Speed_Distance'],
                        'high_accel_count': session['High_Accel_Count'],
                        'high_decel_count': session['High_Decel_Count'],
                        'duration': session['Duration']
                    }
                    pre_match_sessions.append(session_dict)
            
            if not pre_match_sessions:
                continue
                
            # Convert to DataFrame for analysis
            pre_match_df = pd.DataFrame(pre_match_sessions)
            
            # Group by days before match
            taper_strategy = {}
            for days in range(1, 8):
                day_data = pre_match_df[pre_match_df['days_before_match'] == days]
                if len(day_data) >= 2:  # Need at least a couple samples
                    avg_load = player_data[player_data['Session_Type'] == 'training']['Player Load'].mean()
                    avg_hsd = player_data[player_data['Session_Type'] == 'training']['High_Speed_Distance'].mean()
                    
                    taper_strategy[f'MD-{days}'] = {
                        'load_percentage': day_data['player_load'].mean() / avg_load * 100 if avg_load > 0 else 0,
                        'high_intensity': day_data['high_speed_distance'].mean() / avg_hsd * 100 if avg_hsd > 0 else 0,
                        'high_acceleration_count': day_data['high_accel_count'].mean(),
                        'high_deceleration_count': day_data['high_decel_count'].mean(),
                        'recommended_duration': day_data['duration'].median(),
                        'sample_count': len(day_data)  # Add sample count for confidence assessment
                    }
            
            if taper_strategy:
                taper_strategies[player] = taper_strategy
                
        return taper_strategies