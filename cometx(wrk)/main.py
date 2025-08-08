import torch
import numpy as np
import warnings
from src.encoders import get_text_embedding, get_image_embedding
from src.fusion import fuse_embeddings
from src.gnn import build_hetero_graph, GNN
from src.hemmr import explain_prediction
from src.fact_checker import check_facts
from torch_geometric.nn import to_hetero

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.file_download")

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
    print("\n[STEP 1/6] Extracting features from text and image...")
    text_vec = get_text_embedding(sample_post['text_content'])
    image_vec = get_image_embedding(sample_post['image_url'])

    if text_vec is None or image_vec is None:
        print("Feature extraction failed. Exiting.")
        return

    # --- 3. Multimodal Fusion ---
    print("\n[STEP 2/6] Fusing multimodal features...")
    fused_vec = fuse_embeddings(text_vec, image_vec)
    print(f"Fused embedding shape: {fused_vec.shape}")

    # --- 4. Graph Construction (Simulated) ---
    print("\n[STEP 3/6] Building heterogeneous graph...")
    num_posts = 100
    num_users = 50
    post_feature_size = fused_vec.shape[0]
    user_feature_size = 64

    # Placeholder for a full dataset of posts and users
    all_post_features = torch.randn(num_posts, post_feature_size)
    all_post_features[0] = fused_vec
    all_user_features = torch.randn(num_users, user_feature_size)
    
    graph_data = build_hetero_graph(
        num_posts, num_users, all_post_features, all_user_features
    )
    print("Graph built successfully.")

    # --- 5. GNN Inference (Simulated) ---
    print("\n[STEP 4/6] Running GNN inference...")
    gnn_model = GNN(hidden_channels=128, out_channels=2)
    gnn_model = to_hetero(gnn_model, graph_data.metadata(), aggr='sum')
    
    out = gnn_model(graph_data.x_dict, graph_data.edge_index_dict)
    post_prediction = out['post'][0].softmax(dim=-1)
    
    # --- 6. Fact-Checking API Call ---
    print("\n[STEP 5/6] Cross-referencing with fact-checking services...")
    fact_check_result = check_facts(sample_post['text_content'])
    
    # --- 7. HEMMR Explanation ---
    print("\n[STEP 6/6] Generating explanation...")
    attention_weights = {
        'text_focus': "a lot of focus on the words 'miracle cure' and 'all diseases'",
        'image_focus': "a high focus on the unprofessional, text-heavy logo",
        'network_focus': "a high focus on the rapid sharing by newly created user accounts"
    }
    
    explanation = explain_prediction(post_prediction, attention_weights)
    if fact_check_result:
        explanation += (
            f"\n\n***External Fact-Check***\n"
            f"**Claim:** {fact_check_result['claim']}\n"
            f"**Rating:** {fact_check_result['label']} by {fact_check_result['publisher']}\n"
            f"**Summary:** {fact_check_result['summary']}\n"
            f"**More Info:** {fact_check_result['url']}"
        )
    
    print("\n--- FINAL OUTPUT ---")
    print(explanation)

if __name__ == "__main__":
    main()

