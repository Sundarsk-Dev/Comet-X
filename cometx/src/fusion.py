import torch

def fuse_embeddings(text_embedding, image_embedding):
    """
    Combines text and image embeddings via concatenation.
    """
    if text_embedding is None or image_embedding is None:
        return None

    try:
        fused_embedding = torch.cat((text_embedding, image_embedding), dim=0)
        return fused_embedding

    except Exception as e:
        print(f"An error occurred during fusion: {e}")
        return None