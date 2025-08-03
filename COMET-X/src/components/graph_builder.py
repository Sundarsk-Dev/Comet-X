# src/components/graph_builder.py

import pandas as pd
import torch
import numpy as np
import os
from torch_geometric.data import HeteroData
from tqdm import tqdm

# Import paths from config
from src.config import PROCESSED_DATA_DIR, CONTENT_EMBEDDINGS_PKL, GRAPH_DATA_PATH, TEXT_EMBEDDING_DIM, IMAGE_EMBEDDING_DIM


def build_heterogeneous_graph(content_embeddings_path, output_graph_path):
    """
    Builds a PyTorch Geometric HeteroData graph from processed content embeddings.
    For this PoC, it will create 'content' nodes and a dummy 'user' node
    with simple 'posts' edges.
    """
    print(f"Loading content embeddings from: {content_embeddings_path}")
    content_df = pd.read_pickle(content_embeddings_path)

    # --- IMPORTANT: Handle NaN values in 'label_encoded' ---
    initial_rows = len(content_df)
    content_df.dropna(subset=['label_encoded'], inplace=True)
    if len(content_df) < initial_rows:
        print(f"Removed {initial_rows - len(content_df)} rows with NaN labels. Remaining: {len(content_df)}.")
    
    if content_df.empty:
        raise ValueError("No content data available after dropping rows with NaN labels. Cannot build graph.")

    # Initialize HeteroData object
    data = HeteroData()

    # --- Content Nodes ---
    # Convert content_id to numerical indices
    content_id_to_idx = {id: i for i, id in enumerate(content_df['content_id'].unique())}
    data['content'].node_id_map = content_id_to_idx # Store for later lookup

    # Concatenate text, image embeddings, AND AI detection score to form content node features
    # Ensure all components are numpy arrays of float32 and stack them
    text_embeddings = np.stack(content_df['text_embedding'].values).astype(np.float32)
    image_embeddings = np.stack(content_df['image_embedding'].values).astype(np.float32)
    
    # Get the AI image scores and reshape them to be a 2D array for concatenation
    # Each score is a single float, so it becomes (num_samples, 1)
    ai_image_scores = content_df['ai_image_score'].values.astype(np.float32).reshape(-1, 1)
    
    # Convert numpy arrays to torch tensors and concatenate all features
    data['content'].x = torch.tensor(np.concatenate([text_embeddings, image_embeddings, ai_image_scores], axis=1))
    
    # Ensure labels are of type torch.long for CrossEntropyLoss
    data['content'].y = torch.tensor(content_df['label_encoded'].values, dtype=torch.long) # Add labels

    print(f"Created {len(content_id_to_idx)} content nodes with features of shape {data['content'].x.shape}")

    # --- User Nodes (Dummy for PoC) ---
    data['user'].x = torch.zeros((1, 1), dtype=torch.float32) # One dummy user node with a placeholder feature
    user_id_to_idx = {'dummy_user_0': 0}
    data['user'].node_id_map = user_id_to_idx

    print(f"Created {len(user_id_to_idx)} user nodes with features of shape {data['user'].x.shape}")

    # --- Edges: User -> Posts -> Content ---
    src_nodes = [user_id_to_idx['dummy_user_0']] * len(content_df)
    dst_nodes = [content_id_to_idx[cid] for cid in content_df['content_id']]
    data['user', 'posts', 'content'].edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    
    print(f"Created {data['user', 'posts', 'content'].num_edges} 'posts' edges.")

    # Save the graph
    os.makedirs(os.path.dirname(output_graph_path), exist_ok=True)
    torch.save(data, output_graph_path)
    print(f"Graph saved to: {output_graph_path}")

    # Validation
    print("\n--- Graph Statistics ---")
    print(data)
    print(f"Number of content nodes: {data['content'].num_nodes}")
    print(f"Content node feature shape: {data['content'].x.shape}")
    print(f"Number of user nodes: {data['user'].num_nodes}")
    print(f"User node feature shape: {data['user'].x.shape}")
    print(f"Number of 'posts' edges: {data['user', 'posts', 'content'].num_edges}")

    return data

if __name__ == "__main__":
    print("--- Running Graph Builder Locally ---")
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    import sys
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.config import PROCESSED_DATA_DIR, CONTENT_EMBEDDINGS_PKL, GRAPH_DATA_PATH

    if not os.path.exists(CONTENT_EMBEDDINGS_PKL):
        print(f"Error: {CONTENT_EMBEDDINGS_PKL} not found. Please run feature extraction first.")
    else:
        graph = build_heterogeneous_graph(
            content_embeddings_path=CONTENT_EMBEDDINGS_PKL,
            output_graph_path=GRAPH_DATA_PATH
        )
    print("\nGraph building complete.")