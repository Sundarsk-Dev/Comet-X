# src/pipelines/evaluate_model.py
import os
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Import from your project's modules
from src.config import GRAPH_DATA_PATH, MODELS_DIR, GNN_OUTPUT_DIM, GNN_HIDDEN_DIM
from src.models.gnn_architecture import HeteroGNN # Your GNN model definition

def evaluate_gnn_model(graph_path, model_path):
    """
    Loads a trained GNN model and graph data, then evaluates the model's performance.
    """
    print(f"--- Starting Model Evaluation ---")

    # --- Device Setup ---
    # Evaluation is less computationally intensive, can run on CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for evaluation: {device}")

    # --- Load Graph Data ---
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file not found at {graph_path}. Please ensure it's generated and placed correctly.")

    # Load with weights_only=False because the graph was saved with custom PyG objects
    graph = torch.load(graph_path, weights_only=False).to(device)
    print(f"Graph loaded from: {graph_path}")
    print(f"Graph structure: {graph}")

    # --- Load Trained GNN Model ---
    # Create an instance of the model with the same architecture parameters
    model = HeteroGNN(hidden_channels=GNN_HIDDEN_DIM, out_channels=GNN_OUTPUT_DIM).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained GNN model not found at {model_path}. Please ensure it's downloaded from Colab.")

    # Load the saved state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Trained GNN model loaded from: {model_path}")

    model.eval() # Set model to evaluation mode (disables dropout, batch norm updates, etc.)

    # --- Perform Inference ---
    print("\nPerforming inference...")
    with torch.no_grad(): # Disable gradient calculations for inference
        # Pass the dictionary of node features and dictionary of edge indices
        # The output 'out' will contain predictions for 'content' nodes
        out = model(graph.x_dict, graph.edge_index_dict)

        # Get predicted class labels (0 or 1 for binary classification)
        predictions = out.argmax(dim=1)

        # Get true labels from the graph (assuming they are in 'content'.y)
        true_labels = graph['content'].y

    # Convert tensors to numpy arrays for scikit-learn metrics
    predictions_np = predictions.cpu().numpy()
    true_labels_np = true_labels.cpu().numpy()

    # --- Calculate Evaluation Metrics ---
    print("\n--- Evaluation Results ---")

    accuracy = accuracy_score(true_labels_np, predictions_np)
    precision = precision_score(true_labels_np, predictions_np, average='binary') # Use 'binary' for 2 classes
    recall = recall_score(true_labels_np, predictions_np, average='binary')
    f1 = f1_score(true_labels_np, predictions_np, average='binary')

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(true_labels_np, predictions_np)
    print("\nConfusion Matrix:")
    print(cm)
    # For binary classification:
    # [[TN, FP]
    #  [FN, TP]]

    print("\nModel evaluation complete.")
    return accuracy, precision, recall, f1, cm

if __name__ == "__main__":
    # Define paths using config
    # Adjusting project_root calculation for running evaluate_model.py as a script
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    import sys
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.config import GRAPH_DATA_PATH, MODELS_DIR, AI_DETECTOR_MODEL_PATH # AI_DETECTOR_MODEL_PATH is just for MODELS_DIR prefix

    # Path to the trained GNN model
    gnn_model_save_path = os.path.join(MODELS_DIR, 'gnn_models', 'trained_gnn.pth')

    # Run the evaluation
    try:
        evaluate_gnn_model(graph_path=GRAPH_DATA_PATH, model_path=gnn_model_save_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have run the graph builder and GNN training steps, and downloaded the model file.")
    except Exception as e:
        print(f"An unexpected error occurred during evaluation: {e}")