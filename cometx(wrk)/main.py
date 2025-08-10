
import torch
import numpy as np
import warnings
from src.encoders import get_text_embedding, get_image_embedding, detect_ai_generated_image
from src.fusion import fuse_embeddings
from src.gnn import build_hetero_graph, GNN
from src.hemmr import explain_prediction
from src.fact_checker import check_facts
from torch_geometric.nn import to_hetero

# You can add this line at the top to disable the Hugging Face symlink warning
# This is optional but can make the output cleaner
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.file_download")

def main():
    """
    Orchestrates the entire COMET-X pipeline from input to explanation.
    """
    print("--- COMET-X Pipeline Starting ---")

    # --- 1. Data Ingestion (Simulated) ---
    # This is your sample social media post.
    # In a future UI, this data will come from user uploads.
    sample_post = {
        "post_id": "unique_post_id_123",
        "text_content": "A new 'miracle cure' for all diseases is now available.",
        "image_url": "https://placehold.co/224x224/FFAA00/000000/png?text=Fake+Product", # Example: Could be AI-generated looking
        "author_id": "user_456"
    }

    # --- 2. Feature Extraction ---
    print("\n[STEP 1/7] Extracting features from text and image...")
    text_vec = get_text_embedding(sample_post['text_content'])
    image_vec = get_image_embedding(sample_post['image_url'])

    if text_vec is None or image_vec is None:
        print("Feature extraction failed. Exiting.")
        return

    # --- 3. Multimodal Fusion ---
    # Combines the text and image embeddings into a single, comprehensive vector.
    print("\n[STEP 2/7] Fusing multimodal features...")
    fused_vec = fuse_embeddings(text_vec, image_vec)
    print(f"Fused embedding shape: {fused_vec.shape}")

    # --- 4. Graph Construction (Simulated) ---
    # Builds a heterogeneous graph representing posts, users, and their interactions.
    print("\n[STEP 3/7] Building heterogeneous graph...")
    num_posts = 100
    num_users = 50
    post_feature_size = fused_vec.shape[0]
    user_feature_size = 64

    # Placeholder for a full dataset of posts and users
    all_post_features = torch.randn(num_posts, post_feature_size)
    all_post_features[0] = fused_vec # Our sample post is at index 0
    all_user_features = torch.randn(num_users, user_feature_size)
    
    graph_data = build_hetero_graph(
        num_posts, num_users, all_post_features, all_user_features
    )
    print("Graph built successfully.")

    # --- 5. GNN Inference (Simulated) ---
    # Runs the Graph Neural Network to make a prediction based on content and network structure.
    print("\n[STEP 4/7] Running GNN inference...")
    gnn_model = GNN(hidden_channels=128, out_channels=2)
    gnn_model = to_hetero(gnn_model, graph_data.metadata(), aggr='sum')
    
    out = gnn_model(graph_data.x_dict, graph_data.edge_index_dict)
    post_prediction = out['post'][0].softmax(dim=-1)
    
    # --- 6. AI Image Detection ---
    # Uses your custom-trained model to detect if the image is AI-generated.
    print("\n[STEP 5/7] Detecting AI-generated image content...")
    ai_image_detection_result = detect_ai_generated_image(sample_post['image_url'])
    
    # --- 7. Fact-Checking API Call ---
    # Cross-references the text claim with external fact-checking services.
    print("\n[STEP 6/7] Cross-referencing with fact-checking services...")
    fact_check_result = check_facts(sample_post['text_content'])
    
    # --- 8. HEMMR Explanation ---
    # Generates a human-readable explanation combining all analysis results.
    # The hemmr part MUST be the last to execute as it synthesizes all previous steps.
    print("\n[STEP 7/7] Generating explanation...")
    attention_weights = {
        'text_focus': "a lot of focus on the words 'miracle cure' and 'all diseases'",
        'image_focus': "a high focus on the unprofessional, text-heavy logo",
        'network_focus': "a high focus on the rapid sharing by newly created user accounts"
    }
    
    explanation = explain_prediction(post_prediction, attention_weights)
    
    # Add AI Image Detection result to the explanation
    if ai_image_detection_result:
        # A more nuanced message to reflect that a "real" image can still be part of misinformation
        label = ai_image_detection_result['predicted_label']
        confidence = ai_image_detection_result['confidence']
        message = f"This image is classified as '{label}' with {confidence:.2f}% confidence. (Note: A real image can still be used in a misleading context.)" if label == 'Real' else f"This image is classified as '{label}' with {confidence:.2f}% confidence."

        explanation += (
            f"\n\n***AI Image Detection***\n"
            f"**Image Analysis:** {message}"
        )

    # Add External Fact-Check result to the explanation
    if fact_check_result:
        explanation += (
            f"\n\n***External Fact-Check***\n"
            f"**Claim:** {fact_check_result['claim']}\n"
            f"**Rating:** {fact_check_result['label']} by {fact_check_result['publisher']}\n"
            f"**Summary:** {fact_check_result['summary']}\n"
            f"**More Info:** {fact_check_result['url']}"
        )
    else:
        # Custom message for when no relatable facts are found
        explanation += (
            f"\n\n***External Fact-Check***\n"
            f"**Fact-Check Status:** No highly relatable facts found, this claim is more inclined to be fake."
        )
    
    print("\n--- FINAL OUTPUT ---")
    print(explanation)

if __name__ == "__main__":
    main()


