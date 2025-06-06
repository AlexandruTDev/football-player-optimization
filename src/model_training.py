from datetime import timedelta
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance # Keep if needed elsewhere, otherwise remove
from sklearn.metrics import confusion_matrix # Keep if needed elsewhere, otherwise remove
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.neural_network import MLPClassifier


class ModelTrainer:
    """Class responsible for training player performance models."""
    
    def __init__(self):
        """Initialize model training components."""
        self.load_models = {}  # Player-specific models for optimal load
        self.fatigue_models = {'match': {}, 'training': {}}  # Separate models for match/training
        self.taper_models = {}  # Player-specific models for tapering strategies
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

    def _train_player_fatigue_model(self, data, session_type):
        """Enhanced fatigue prediction with multiple model comparison and better evaluation"""
        
        players = data['Player Name'].unique()
        
        # Train separate models for match and training fatigue
        self.fatigue_models[session_type] = {}
        
        # Train a separate model for each player and session type
        for player in players:
            player_data = data[data['Player Name'] == player].sort_values('Date').copy() # Use copy to avoid SettingWithCopyWarning
            type_data = player_data[player_data['Session_Type'] == session_type].copy() # Use copy
            if len(type_data) < 20:  # Need sufficient data
                continue

            # Define features for fatigue prediction
            features = ['ACWR', 'Load_Monotony', 'Load_Strain', 'High_Speed_Distance', 
                        'High_Accel_Count', 'High_Decel_Count', 'High_Impact_Count',
                        'Days_Since_Match']
            
            if session_type == 'training':
                # Add training-specific features
                features.extend(['Training_ACWR', 'Days_Until_Match'])
            
            # Target: Speed decline in next session of same type
            target_col = f'{session_type.capitalize()}_Speed_Decline'
            if target_col not in type_data.columns:
                continue

            type_data[f'Next_{session_type.capitalize()}_Speed_Decline'] = type_data[f'{session_type.capitalize()}_Speed_Decline'].shift(-1)
            type_data['Performance_Drop'] = (type_data[f'Next_{session_type.capitalize()}_Speed_Decline'] < -0.05).astype(int)
            
            # Prepare training data
            X = type_data[features].dropna()
            y = type_data['Performance_Drop'].loc[X.index].dropna()
            
            # Make sure X and y have same indices
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]
            
            if len(X) < 15:  # Not enough samples after cleaning
                continue
            
            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Define models to compare
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'GradientBoosting': GradientBoostingClassifier(random_state=42),
                'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(50,20), max_iter=1000, random_state=42)
            }
            
            # Use GroupKFold to prevent data leakage due to time-series nature
            groups = type_data.loc[common_idx]['Date'].dt.strftime('%Y-%m')  # Group by month
            
            best_model = None
            best_score = 0
            best_model_name = None
            
            for name, model in models.items():
                # Cross-validation with GroupKFold to respect time order
                if len(np.unique(y)) < 2:
                    # Skip cross-validation if only one class is present in the target
                    avg_score = 0
                else:
                    if len(np.unique(groups)) >= 3:  # Need at least 3 time periods for cross-validation
                        cv = GroupKFold(n_splits=min(3, len(np.unique(groups))))
                        try:
                            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc', groups=groups)
                        except ValueError:
                            # Handle case where a fold has only one class
                            scores = [0] # Assign a low score
                        avg_score = np.mean(scores)
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_model_name = name
                        best_model = model
        
            if best_model is not None and best_score > 0.65:  # Higher threshold for model quality
                # Retrain on full dataset
                if len(np.unique(y)) >= 2:
                    # Calculate confusion matrix before fitting again (optional, but good practice)
                    # y_pred = best_model.predict(X_scaled) # Prediction before potential re-fitting
                    # cm = confusion_matrix(y, y_pred) # CM before potential re-fitting

                    best_model.fit(X_scaled, y)

                    y_pred = best_model.predict(X_scaled) # Prediction after re-fitting
                    cm = confusion_matrix(y, y_pred)

                    best_model.fit(X_scaled, y)
                else:
                    best_model = None # Cannot train a meaningful model with only one class
                
                # Calculate feature importances if available
                if hasattr(best_model, 'feature_importances_'):
                    importances = best_model.feature_importances_
                    feature_importance = dict(zip(features, importances))
                else:
                    feature_importance = None
                
                self.fatigue_models[session_type][player] = {
                    'model': best_model,
                    'model_type': best_model_name,
                    'auc_score': best_score,
                    'confusion_matrix': cm.tolist() if 'cm' in locals() else None, # Store as list
 'features': features,
                    'feature_importance': feature_importance
                }

    def train_fatigue_prediction_model(self, data):
        """Train models to predict performance declines based on accumulated fatigue."""
        players = data['Player Name'].unique()
        
        # Train separate models for match and training fatigue
        for session_type in self.session_types:
            self.fatigue_models[session_type] = {}

            # The _train_player_fatigue_model function now handles player iteration
            self._train_player_fatigue_model(data, session_type)

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