import joblib
from matplotlib import pyplot as plt
import numpy as np

from dashboard_generation import DashboardGenerator
from data_preprocessing import DataPreprocessor
from insight_generation import InsightGenerator
from model_training import ModelTrainer


class PlayerPerformanceOptimizer:
    """Main class that coordinates the optimization workflow."""
    
    def __init__(self):
        """Initialize optimizer components."""
        self.preprocessor = DataPreprocessor()
        self.model_trainer = ModelTrainer()
        self.insight_generator = None  # Will be initialized after training
        self.dashboard_generator = DashboardGenerator()
        
        # Model storage
        self.optimal_zones = {}
        self.fatigue_models = {'match': {}, 'training': {}}
        self.taper_strategies = {}
        
        # Initialization flags
        self.is_trained = False
    
    def fit(self, training_data):
        """Main method to train all models using historical data."""
        print(f"Processing data for {len(training_data['Player Name'].unique())} players...")
        
        # Step 1: Preprocess data
        processed_data = self.preprocessor.process_data(training_data)
        
        if processed_data.empty:
            print("Warning: No players had sufficient data after preprocessing")
            return self
            
        # Step 2: Identify optimal load zones
        print("Identifying optimal load zones based on match performances...")
        self.optimal_zones = self.model_trainer.identify_optimal_load_zones(processed_data)
        print(f"Found optimal zones for {len(self.optimal_zones)} players")
        
        # Step 3: Train fatigue prediction models
        print("Training fatigue prediction models for match and training sessions...")
        self.model_trainer.train_fatigue_prediction_model(processed_data)
        self.fatigue_models = self.model_trainer.fatigue_models
        print(f"Created match fatigue models for {len(self.fatigue_models['match'])} players")
        print(f"Created training fatigue models for {len(self.fatigue_models['training'])} players")
        
        # Step 4: Develop tapering strategies
        print("Developing tapering strategies...")
        self.taper_strategies = self.model_trainer.develop_tapering_strategies(processed_data)
        print(f"Created tapering strategies for {len(self.taper_strategies)} players")
        
        # Initialize insight generator with trained models
        self.insight_generator = InsightGenerator(
            self.optimal_zones,
            self.fatigue_models,
            self.taper_strategies
        )
        
        self.is_trained = True
        return self
    
    def predict(self, new_data):
        """Generate insights for all players in new data."""
        if not self.is_trained:
            return {"error": "Model has not been trained yet. Call fit() first."}
            
        # Preprocess the new data
        processed_new_data = self.preprocessor.process_data(new_data)
        
        if processed_new_data.empty:
            return {"error": "No valid player data after preprocessing"}
            
        # Get unique players in the new data
        players = processed_new_data['Player Name'].unique()
        all_insights = {}
        
        for player in players:
            player_insights = self.insight_generator.generate_player_insights(player, processed_new_data)
            if player_insights:
                all_insights[player] = player_insights
                
        return all_insights
    
    def generate_dashboard_data(self, insights):
        """Create visualization-ready data for dashboards."""
        return self.dashboard_generator.generate_dashboard_data(insights)
    
    def save_model(self, filepath):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
            
        model_data = {
            'optimal_zones': self.optimal_zones,
            'fatigue_models': self.fatigue_models,
            'taper_strategies': self.taper_strategies
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath):
        """Load trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        model_data = joblib.load(filepath)
        
        self.optimal_zones = model_data['optimal_zones']
        self.fatigue_models = model_data['fatigue_models']
        self.taper_strategies = model_data['taper_strategies']
        
        # Initialize insight generator with loaded models
        self.insight_generator = InsightGenerator(
            self.optimal_zones,
            self.fatigue_models,
            self.taper_strategies
        )
        
        self.is_trained = True
        print(f"Model loaded from {filepath}")
        return self
        
    def visualize_insights(self, insights, player=None):
        """Generate visualizations for player insights."""
        if player is not None and player in insights:
            return self._visualize_player_insights(player, insights[player])
        else:
            # Create team overview visualizations
            return self._visualize_team_insights(insights)
    
    def _visualize_player_insights(self, player_name, player_insights):
        """Generate visualizations for a specific player."""
        visualizations = {}
        
        # Skip players with errors
        if 'error' in player_insights:
            return {"error": player_insights['error']}
            
        # Create fatigue risk visualization
        if 'fatigue_prediction' in player_insights:
            fig, ax = plt.subplots(figsize=(8, 4))
            risk_score = player_insights['fatigue_prediction']['risk_score']
            
            # Create gauge chart
            ax.axis('equal')
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.add_patch(plt.Circle((5, 5), 4, fc='#f0f0f0', ec='gray'))
            
            # Add colored segments
            wedge_green = plt.matplotlib.patches.Wedge((5, 5), 4, -30, 30, fc='green', alpha=0.7)
            wedge_yellow = plt.matplotlib.patches.Wedge((5, 5), 4, 30, 90, fc='yellow', alpha=0.7)
            wedge_orange = plt.matplotlib.patches.Wedge((5, 5), 4, 90, 150, fc='orange', alpha=0.7)
            wedge_red = plt.matplotlib.patches.Wedge((5, 5), 4, 150, 210, fc='red', alpha=0.7)
            
            ax.add_patch(wedge_green)
            ax.add_patch(wedge_yellow)
            ax.add_patch(wedge_orange)
            ax.add_patch(wedge_red)
            
            # Add risk indicator needle
            angle = -30 + 240 * (risk_score / 100)
            x = 5 + 3.5 * np.cos(np.radians(angle))
            y = 5 + 3.5 * np.sin(np.radians(angle))
            ax.plot([5, x], [5, y], color='black', linewidth=2)
            
            # Add central circle
            ax.add_patch(plt.Circle((5, 5), 0.3, fc='black'))
            
            # Add labels
            ax.text(5, 1, f"Risk Score: {risk_score:.1f}%", ha='center', fontsize=12)
            ax.text(5, 9, f"{player_name} - Fatigue Risk", ha='center', fontsize=14, fontweight='bold')
            ax.text(5, 0.5, f"Confidence: {player_insights['fatigue_prediction'].get('confidence', 0):.1f}%", 
                   ha='center', fontsize=10)
            
            # Remove axis ticks
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add key factors if available
            if 'key_factors' in player_insights['fatigue_prediction']:
                factors = player_insights['fatigue_prediction']['key_factors']
                factor_text = ""
                for i, factor in enumerate(factors):
                    factor_text += f"{i+1}. {factor['feature']}: {factor['status']} ({factor['importance']:.1f}%)\n"
                
                ax.text(9.5, 5, "Key Factors:", ha='left', fontsize=10, fontweight='bold')
                ax.text(9.5, 4.5, factor_text, ha='left', va='top', fontsize=9)
            
            plt.tight_layout()
            visualizations['fatigue_gauge'] = fig
            
        # Create tapering strategy visualization if available
        if 'taper_strategy' in player_insights:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            days = []
            load_percentages = []
            intensity_percentages = []
            
            for day in range(1, 8):
                day_key = f"MD-{day}"
                if day_key in player_insights['taper_strategy']:
                    days.append(f"MD-{day}")
                    load_percentages.append(player_insights['taper_strategy'][day_key]['load_percentage'])
                    intensity_percentages.append(player_insights['taper_strategy'][day_key]['high_intensity'])
            
            if days:
                # Reverse lists to have MD-1 on the right
                days.reverse()
                load_percentages.reverse()
                intensity_percentages.reverse()
                
                x = np.arange(len(days))
                width = 0.35
                
                # Plot bars
                rects1 = ax.bar(x - width/2, load_percentages, width, label='Load %', color='blue', alpha=0.7)
                rects2 = ax.bar(x + width/2, intensity_percentages, width, label='High Intensity %', color='red', alpha=0.7)
                
                # Add labels
                ax.set_ylabel('Percentage of Average (%)')
                ax.set_title(f'{player_name} - Optimal Tapering Strategy', fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(days)
                ax.legend()
                
                # Add value labels on bars
                def autolabel(rects):
                    for rect in rects:
                        height = rect.get_height()
                        ax.annotate(f'{height:.0f}%',
                                    xy=(rect.get_x() + rect.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=8)
                
                autolabel(rects1)
                autolabel(rects2)
                
                # Add reference line at 100%
                ax.axhline(y=100, color='gray', linestyle='--', alpha=0.7)
                
                # Highlight today if available
                if 'days_until_next_match' in player_insights:
                    days_until = player_insights['days_until_next_match']
                    if 1 <= days_until <= 7:
                        day_index = days.index(f"MD-{days_until}") if f"MD-{days_until}" in days else None
                        if day_index is not None:
                            # Add highlight for today
                            ax.axvspan(day_index - 0.5, day_index + 0.5, alpha=0.2, color='yellow')
                            ax.text(day_index, 10, "TODAY", ha='center', fontweight='bold')
                
                plt.tight_layout()
                visualizations['taper_strategy'] = fig
        
        return visualizations
    
    def _visualize_team_insights(self, insights):
        """Generate team-level visualizations."""
        visualizations = {}
        
        # Create fatigue risk heatmap
        fatigue_data = []
        players = []
        risk_scores = []
        
        for player, player_insights in insights.items():
            if 'fatigue_prediction' in player_insights:
                players.append(player)
                risk_scores.append(player_insights['fatigue_prediction']['risk_score'])
        
        if players:
            # Sort by risk score
            sorted_indices = np.argsort(risk_scores)[::-1]  # Descending order
            sorted_players = [players[i] for i in sorted_indices]
            sorted_risks = [risk_scores[i] for i in sorted_indices]
            
            fig, ax = plt.subplots(figsize=(12, max(6, len(players) * 0.4)))
            
            # Create horizontal bar chart
            bars = ax.barh(sorted_players, sorted_risks, height=0.7)
            
            # Color bars based on risk level
            for i, bar in enumerate(bars):
                if sorted_risks[i] > 70:
                    bar.set_color('red')
                elif sorted_risks[i] > 50:
                    bar.set_color('orange')
                elif sorted_risks[i] > 30:
                    bar.set_color('yellow')
                else:
                    bar.set_color('green')
            
            # Add labels and title
            ax.set_xlabel('Fatigue Risk Score (%)')
            ax.set_title('Team Fatigue Risk Assessment', fontweight='bold')
            
            # Add grid lines
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(sorted_risks):
                ax.text(v + 1, i, f"{v:.1f}%", va='center')
            
            # Add risk level guidelines
            ax.axvline(x=30, color='green', linestyle='--', alpha=0.5)
            ax.axvline(x=50, color='yellow', linestyle='--', alpha=0.5)
            ax.axvline(x=70, color='red', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            visualizations['team_fatigue'] = fig
        
        # Create player readiness overview
        readiness_data = {'optimal': 0, 'underloaded': 0, 'overloaded': 0, 'unknown': 0}
        
        for player, player_insights in insights.items():
            if 'load_status' in player_insights:
                status = player_insights['load_status'].get('status', 'UNKNOWN').lower()
                readiness_data[status] += 1
            else:
                readiness_data['unknown'] += 1
        
        if sum(readiness_data.values()) > 0:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Create pie chart
            colors = {'optimal': 'green', 'underloaded': 'yellow', 'overloaded': 'red', 'unknown': 'gray'}
            labels = [f"{k.capitalize()} ({v})" for k, v in readiness_data.items() if v > 0]
            sizes = [v for v in readiness_data.values() if v > 0]
            ax_colors = [colors[k] for k in readiness_data.keys() if readiness_data[k] > 0]
            
            wedges, texts, autotexts = ax.pie(
                sizes, 
                labels=labels, 
                colors=ax_colors,
                autopct='%1.1f%%', 
                startangle=90,
                explode=[0.05] * len(sizes)
            )
            
            # Style the labels and percentages
            for text in texts:
                text.set_fontsize(12)
            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_fontweight('bold')
                
            ax.axis('equal')
            ax.set_title('Team Readiness Status Overview', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            visualizations['team_readiness'] = fig
        
        return visualizations