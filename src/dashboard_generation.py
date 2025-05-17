class DashboardGenerator:
    """Class responsible for creating visualization-ready data for dashboards."""
    
    def generate_dashboard_data(self, insights):
        """Create visualization-ready data for dashboards."""
        dashboard_data = {
            'player_status': [],
            'fatigue_risk': [],
            'taper_recommendations': [],
            'match_preparation': []
        }
        
        for player, player_insights in insights.items():
            # Skip players with errors
            if 'error' in player_insights:
                continue
                
            # Session type indicator
            session_type = player_insights.get('session_type', 'unknown')
            
            # Player status for traffic light system
            status = "unknown"
            if 'load_status' in player_insights:
                status = player_insights['load_status'].get('status', 'unknown').lower()
                    
            dashboard_data['player_status'].append({
                'player': player,
                'status': status,
                'session_type': session_type,
                'message': player_insights.get('load_status', {}).get('message', '')
            })
            
            # Fatigue risk scores
            if 'fatigue_prediction' in player_insights:
                fatigue_data = {
                    'player': player,
                    'risk_score': player_insights['fatigue_prediction']['risk_score'],
                    'interpretation': player_insights['fatigue_prediction']['interpretation'],
                    'session_type': session_type,
                    'confidence': player_insights['fatigue_prediction']['confidence']
                }
                
                # Add key factors if available
                if 'key_factors' in player_insights['fatigue_prediction']:
                    fatigue_data['key_factors'] = player_insights['fatigue_prediction']['key_factors']
                    
                dashboard_data['fatigue_risk'].append(fatigue_data)
                
            # Match preparation - for players with upcoming matches
            if 'days_until_next_match' in player_insights and player_insights['days_until_next_match'] <= 7:
                dashboard_data['match_preparation'].append({
                    'player': player,
                    'days_until_match': player_insights['days_until_next_match'],
                    'load_status': status
                })
                
                # Taper recommendations
                if 'today_recommended' in player_insights:
                    taper_data = {
                        'player': player,
                        'days_before_match': player_insights['days_until_next_match'],
                        'load_percentage': player_insights['today_recommended']['load_target'],
                        'high_intensity': player_insights['today_recommended']['high_intensity_target'],
                        'duration': player_insights['today_recommended']['duration_target'],
                        'confidence': player_insights['today_recommended'].get('confidence', 50)
                    }
                    
                    # Add actual vs target comparison if available
                    if 'today_comparison' in player_insights:
                        taper_data.update({
                            'actual_load': player_insights['today_comparison']['actual_load_pct'],
                            'load_difference': player_insights['today_comparison']['load_difference'],
                            'within_target': player_insights['today_comparison']['within_target']
                        })
                        
                    dashboard_data['taper_recommendations'].append(taper_data)
        
        return dashboard_data