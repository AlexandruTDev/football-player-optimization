import os

from generate_insights import generate_insights
from train_model import train_model

def run_pipeline(data_path="data/raw/report_export.csv", player_name=None):
    """Run the complete optimization pipeline."""
    # Train model
    optimizer = train_model()
    
    # Generate insights (using same data for simplicity)
    insights = generate_insights("models/player_optimization_model.joblib", data_path, player_name)
    
    print(f"Pipeline complete! Visualizations saved to 'visualizations' directory.")
    
    return optimizer, insights

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run football player optimization pipeline")
    parser.add_argument("--data", default="data/raw/report_export.csv", 
                        help="Path to the data CSV file")
    parser.add_argument("--player", help="Focus on a specific player")
    
    args = parser.parse_args()
    
    run_pipeline(args.data, args.player)