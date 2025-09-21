import os
import pandas as pd

from .generate_insights import generate_insights
from .data_preprocessing import split_data, preprocess_for_training # Assuming preprocess_for_training exists
from .prepare_data import load_and_preprocess_data # Assuming this function exists

def run_pipeline(data_path="/data/raw/report_export.csv", player_name=None):
    """Run the complete optimization pipeline."""
    # Load and preprocess data
    processed_df = load_and_preprocess_data(data_path) # Assuming this function handles initial loading and preprocessing

    # Split data into training and testing sets
    # Assuming 'target' is the name of your target variable
    if 'target' not in processed_df.columns:
        raise ValueError("Target column 'target' not found in the processed data.")
    X_train, X_test, y_train, y_test = split_data(processed_df)
    
    # Prepare data for training (e.g., scaling, handling missing values)
    # Assuming preprocess_for_training takes X_train and X_test and returns processed versions
    X_train_processed, X_test_processed = preprocess_for_training(X_train, X_test)

    # Train model
    optimizer = train_model(X_train, y_train) # Pass training data to train_model

    # Generate insights (using same data for simplicity)
    insights = generate_insights("/models/player_optimization_model.joblib", "/data/raw/report_export.csv", player_name)

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