import os

from matplotlib import pyplot as plt

from main import PlayerPerformanceOptimizer
from prepare_data import load_and_prepare_data

def generate_insights(model_path="models/player_optimization_model.joblib", 
                     new_data_path="data/raw/new_data.csv",
                     player_name=None):
    """Generate insights from trained model."""
    # Load model
    print(f"Loading model from {model_path}...")
    optimizer = PlayerPerformanceOptimizer()
    optimizer.load_model(model_path)
    
    # Load new data
    print(f"Loading new data from {new_data_path}...")
    new_data = load_and_prepare_data(new_data_path)
    
    # Generate insights
    print("Generating insights...")
    insights = optimizer.predict(new_data)
    
    # Create visualizations directory
    os.makedirs("visualizations", exist_ok=True)
    
    # Generate visualizations
    if player_name and player_name in insights:
        print(f"Generating visualizations for player: {player_name}")
        visualizations = optimizer.visualize_insights(insights, player_name)
        
        # Save visualizations
        for name, fig in visualizations.items():
            fig.savefig(f"visualizations/{player_name}_{name}.png")
            plt.close(fig)
    else:
        print("Generating team-level visualizations")
        visualizations = optimizer.visualize_insights(insights)
        
        # Save visualizations
        for name, fig in visualizations.items():
            fig.savefig(f"visualizations/team_{name}.png")
            plt.close(fig)
    
    print("Insight generation complete!")
    
    return insights

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate player insights")
    parser.add_argument("--model", default="models/player_optimization_model.joblib", 
                        help="Path to the saved model")
    parser.add_argument("--data", default="data/raw/report_export.csv", 
                        help="Path to the new data CSV file")
    parser.add_argument("--player", help="Focus on a specific player")
    
    args = parser.parse_args()
    
    generate_insights(args.model, args.data, args.player)