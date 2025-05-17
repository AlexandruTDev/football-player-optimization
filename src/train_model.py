import pandas as pd
import os
from main import PlayerPerformanceOptimizer
from prepare_data import load_and_prepare_data

def train_model():
    """Train the player performance optimization model."""
    # Load and prepare data
    data_path = "data/raw/report_export.csv"
    print(f"Loading data from {data_path}...")
    training_data = load_and_prepare_data(data_path)
    
    # Create optimizer
    print("Initializing PlayerPerformanceOptimizer...")
    optimizer = PlayerPerformanceOptimizer()
    
    # Train model
    print("Training model...")
    optimizer.fit(training_data)
    
    # Save trained model
    model_path = "models/player_optimization_model.joblib"
    print(f"Saving model to {model_path}...")
    os.makedirs("models", exist_ok=True)
    optimizer.save_model(model_path)
    
    print("Training complete!")
    
    return optimizer

if __name__ == "__main__":
    train_model()