import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(true_labels, predicted_labels, player="", session_type=""):
    """ 
    Generates and saves a confusion matrix visualization.

    Args:
        true_labels (list or array): The true labels.
        predicted_labels (list or array): The predicted labels.
        output_path (str): The path to save the confusion matrix image.
 player (str, optional): The name of the player. Defaults to "".
 session_type (str, optional): The type of session. Defaults to "".
    """
    output_path = f"visualizations/model_performance/confusion_matrix_{player}_{session_type}.png"
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    title = 'Confusion Matrix'
    if player or session_type:
        title += f' for {player} ({session_type})'
    plt.title(title)
    plt.savefig(output_path)
    plt.close()

if __name__ == '__main__':
    # Example usage:
    # Generate some sample data
    true_labels_sample = np.random.randint(0, 2, 100)  # Binary classification example
    predicted_labels_sample = np.random.randint(0, 2, 100)

    # Generate and save the confusion matrix
    plot_confusion_matrix(true_labels_sample, predicted_labels_sample)
    print(f"Confusion matrix saved to confusion_matrix.png")