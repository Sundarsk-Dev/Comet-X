import torch
import numpy as np
from src.encoders import get_text_embedding, get_image_embedding
from src.fusion import fuse_embeddings
from src.gnn import build_hetero_graph, GNN
from src.hemmr import explain_prediction
from torch_geometric.nn import to_hetero

def main():
    """
    Orchestrates the entire COMET-X pipeline from input to explanation.
    """
    print("--- COMET-X Pipeline Starting ---")

    # --- 1. Data Ingestion (Simulated) ---
    sample_post = {
        "post_id": "unique_post_id_123",
        "text_content": "A new 'miracle cure' for all diseases is now available.",
        "image_url": "https://placehold.co/224x224/FFAA00/000000/png?text=Fake+Product",
        "author_id": "user_456"
    }

    # --- 2. Feature Extraction ---
    print("\n[STEP 1/5] Extracting features from text and image...")
    text_vec = get_text_embedding(sample_post['text_content'])
    image_vec = get_image_embedding(sample_post['image_url'])

    if text_vec is None or image_vec is None:
        print("Feature extraction failed. Exiting.")
        return

    # --- 3. Multimodal Fusion ---
    print("\n[STEP 2/5] Fusing multimodal features...")
    fused_vec = fuse_embeddings(text_vec, image_vec)
    print(f"Fused embedding shape: {fused_vec.shape}")

    # --- 4. Graph Construction (Simulated) ---
    print("\n[STEP 3/5] Building heterogeneous graph...")
    num_posts = 100
    num_users = 50
    post_feature_size = fused_vec.shape[0]
    user_feature_size = 64

    # Placeholder for a full dataset of posts and users
    all_post_features = torch.randn(num_posts, post_feature_size)
    all_post_features[0] = fused_vec # Our sample post is at index 0
    all_user_features = torch.randn(num_users, user_feature_size)
    
    # Placeholder for edges
    user_posts_edges = torch.randint(0, num_posts, (2, 20))
    
    graph_data = build_hetero_graph(
        num_posts, num_users, all_post_features, all_user_features, user_posts_edges
    )
    print("Graph built successfully.")

    # --- 5. GNN Inference (Simulated) ---
    print("\n[STEP 4/5] Running GNN inference...")
    gnn_model = GNN(hidden_channels=128, out_channels=2)
    gnn_model = to_hetero(gnn_model, graph_data.metadata(), aggr='sum')
    
    # We will use a placeholder for a pre-trained model's state dictionary here
    # For now, we will just use the randomly initialized model
    out = gnn_model(graph_data.x_dict, graph_data.edge_index_dict)
    
    # Get the prediction for our sample post (which is at index 0)
    post_prediction = out['post'][0].softmax(dim=-1)
    
    # --- 6. HEMMR Explanation ---
    print("\n[STEP 5/5] Generating explanation...")
    # These are placeholder attention weights for the explanation
    attention_weights = {
        'text_focus': "a lot of focus on the words 'miracle cure' and 'all diseases'",
        'image_focus': "a high focus on the unprofessional, text-heavy logo",
        'network_focus': "a high focus on the rapid sharing by newly created user accounts"
    }
    
    explanation = explain_prediction(post_prediction, attention_weights)
    print("\n--- FINAL OUTPUT ---")
    print(explanation)

if __name__ == "__main__":
    # Ensure all required libraries are installed
    # The following commands will only work if you are in a Colab environment.
    # In VS Code, you'd use 'pip install -r requirements.txt' from your terminal.
    # !pip install transformers torch numpy torchvision Pillow torch_geometric
    main()

