class InsightGenerator:
    """Class responsible for generating player insights."""
    
    def __init__(self, optimal_zones, fatigue_models, taper_strategies):
        """Initialize with trained models."""
        self.optimal_zones = optimal_zones
        self.fatigue_models = fatigue_models
        self.taper_strategies = taper_strategies
    
    def generate_player_insights(self, player_name, new_data):
        """Generate performance optimization insights for a specific player."""
        # Extract only this player's data
        player_data = new_data[new_data['Player Name'] == player_name]
        
        if player_data.empty:
            return {"error": f"No data found for player {player_name}"}
            
        # Determine latest session type
        latest_session = player_data.sort_values('Date').iloc[-1]
        session_type = latest_session['Session_Type']
        
        insights = {
            'session_type': session_type,
            'latest_date': latest_session['Date'].strftime('%Y-%m-%d'),
            'player_info': {
                'name': player_name,
                'position': latest_session.get('Position', 'Unknown') if 'Position' in latest_session else 'Unknown'
            }
        }
        
        # Get next match information
        if 'Days_Until_Match' in latest_session and not pd.isna(latest_session['Days_Until_Match']):
            insights['days_until_next_match'] = int(latest_session['Days_Until_Match'])
        
        # Get player's optimal load zones if available
        if player_name in self.optimal_zones:
            player_zones = self.optimal_zones[player_name]
            
            # Get zones specific to this session type
            if session_type in player_zones:
                insights['optimal_zones'] = player_zones[session_type]
                
                # For training sessions, check if we're in match preparation period
                if session_type == 'training' and 'days_until_next_match' in insights:
                    days_until = insights['days_until_next_match']
                    if days_until <= 7:
                        md_key = f'MD-{days_until}'
                        if md_key in player_zones['training']:
                            insights['match_prep_targets'] = player_zones['training'][md_key]
                            
                            # Calculate if current load is within optimal zone
                            current_load = latest_session['Player Load']
                            optimal_min, optimal_max = insights['match_prep_targets'].get('daily_load', (0, 0))
                            
                            if current_load < optimal_min:
                                insights['load_status'] = {
                                    "status": "UNDERLOADED",
                                    "message": "Current load below optimal zone for match preparation",
                                    "adjustment": f"Consider increasing load by {((optimal_min - current_load) / current_load * 100):.1f}%"
                                }
                            elif current_load > optimal_max:
                                insights['load_status'] = {
                                    "status": "OVERLOADED",
                                    "message": "Current load above optimal zone for match preparation",
                                    "adjustment": f"Consider decreasing load by {((current_load - optimal_max) / current_load * 100):.1f}%"
                                }
                            else:
                                insights['load_status'] = {
                                    "status": "OPTIMAL",
                                    "message": "Current load within optimal zone for match preparation"
                                }
            else:
                # General assessment without session-specific data
                insights['load_status'] = {
                    "status": "UNKNOWN",
                    "message": "Insufficient historical data for this session type"
                }
        
        # Predict fatigue if model available
        if player_name in self.fatigue_models.get(session_type, {}):
            model_info = self.fatigue_models[session_type][player_name]
            model = model_info['model']
            features = model_info['features']
            
            # Ensure all features are available and handle missing values
            feature_data = latest_session[features].values.reshape(1, -1) if all(f in latest_session for f in features) else None
            
            if feature_data is not None and not np.isnan(feature_data).any():
                fatigue_risk = model.predict_proba(feature_data)[0][1]  # Probability of performance drop
                insights['fatigue_prediction'] = {
                    'risk_score': fatigue_risk * 100,
                    'interpretation': 'HIGH RISK' if fatigue_risk > 0.7 else 'MODERATE RISK' if fatigue_risk > 0.3 else 'LOW RISK',
                    'confidence': model_info['accuracy'] * 100,
                    'key_factors': self._get_key_fatigue_factors(model_info, latest_session)
                }
        

        # Get tapering strategy if available and near a match
            if 'days_until_next_match' in insights and insights['days_until_next_match'] <= 7:
                if player_name in self.taper_strategies:
                    insights['taper_strategy'] = self.taper_strategies[player_name]
                    
                    # Add specific recommendations for today
                    days_key = f"MD-{insights['days_until_next_match']}"
                    if days_key in self.taper_strategies[player_name]:
                        today_strategy = self.taper_strategies[player_name][days_key]
                        insights['today_recommended'] = {
                            'load_target': today_strategy['load_percentage'],
                            'high_intensity_target': today_strategy['high_intensity'],
                            'duration_target': today_strategy['recommended_duration'],
                            'confidence': min(100, today_strategy.get('sample_count', 1) * 20)  # Confidence based on sample size
                        }
                        
                        # Calculate how today's session compares to recommendation
                        if 'Player Load' in latest_session:
                            avg_load = player_data[player_data['Session_Type'] == 'training']['Player Load'].mean()
                            today_load_pct = latest_session['Player Load'] / avg_load * 100 if avg_load > 0 else 0
                            
                            insights['today_comparison'] = {
                                'actual_load_pct': today_load_pct,
                                'load_difference': today_load_pct - today_strategy['load_percentage'],
                                'within_target': abs(today_load_pct - today_strategy['load_percentage']) < 15
                            }
        
        return insights
    
    def _get_key_fatigue_factors(self, model_info, latest_session):
        """Extract key factors contributing to fatigue prediction."""
        if 'feature_importance' not in model_info:
            return []
            
        # Get top 3 most important features
        top_features = sorted(model_info['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:3]
        key_factors = []
        
        for feature, importance in top_features:
            if feature in latest_session:
                value = latest_session[feature]
                
                # Add context to the factor based on feature type
                if 'ACWR' in feature:
                    if value > 1.5:
                        status = "HIGH"
                        impact = "increasing fatigue risk"
                    elif value < 0.8:
                        status = "LOW"
                        impact = "potentially insufficient stimulus"
                    else:
                        status = "OPTIMAL"
                        impact = "well-balanced workload"
                elif 'Load_Monotony' in feature:
                    if value > 2:
                        status = "HIGH"
                        impact = "training lacks variability"
                    else:
                        status = "OPTIMAL"
                        impact = "good training variation"
                elif 'High_Speed_Distance' in feature or 'High_Accel_Count' in feature or 'High_Decel_Count' in feature:
                    # Would need historical data for player to determine thresholds
                    status = "PRESENT"
                    impact = "contributing to fatigue accumulation"
                elif 'Days_Since_Match' in feature:
                    if value < 3:
                        status = "RECENT"
                        impact = "insufficient recovery time"
                    else:
                        status = "SUFFICIENT"
                        impact = "adequate recovery period"
                else:
                    status = "NOTEWORTHY"
                    impact = "influencing fatigue levels"
                
                key_factors.append({
                    'feature': feature,
                    'importance': importance * 100,  # Convert to percentage
                    'value': value,
                    'status': status,
                    'impact': impact
                })
                
        return key_factors