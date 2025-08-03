# src/components/feature_extraction.py

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
from torchvision import transforms
import os
import cv2 # For video processing if you implement it

# Import your AI Detector model
from src.models.ai_detector_model import AIGeneratedDetector

# Import paths and dimensions from config
from src.config import (
    PROCESSED_DATA_DIR, INTERIM_DATA_DIR,
    CONTENT_METADATA_PKL, CONTENT_EMBEDDINGS_PKL,
    TEXT_EMBEDDING_MODEL, IMAGE_EMBEDDING_MODEL,
    TEXT_EMBEDDING_DIM, IMAGE_EMBEDDING_DIM,
    AI_DETECTOR_MODEL_PATH # Ensure this is defined in src/config.py
)

# --- Initialize Models (Load once to avoid re-loading for each image/text) ---
# Text Embedding Model
try:
    text_tokenizer = AutoTokenizer.from_pretrained(TEXT_EMBEDDING_MODEL)
    text_model = AutoModel.from_pretrained(TEXT_EMBEDDING_MODEL)
    text_model.eval() # Set to evaluation mode
    print(f"Loaded text embedding model: {TEXT_EMBEDDING_MODEL}")
except Exception as e:
    print(f"Error loading text embedding model: {e}")
    text_tokenizer, text_model = None, None # Set to None if loading fails

# Image Embedding Model (e.g., ResNet from torchvision)
try:
    # We'll use a pre-trained ResNet for image embeddings, similar to the AI Detector's backbone
    image_embedding_model = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Here, we need to extract features, not classify. So, we'll load a model
        # and remove its final classification layer.
        models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    ])
    # Remove the final classification layer to get features
    # ResNet-18's classification layer is 'fc'
    image_embedding_model.fc = torch.nn.Identity() # Replace 'fc' with an identity layer
    image_embedding_model.eval() # Set to evaluation mode
    print(f"Loaded image embedding model (ResNet-18 features).")
except Exception as e:
    print(f"Error loading image embedding model: {e}")
    image_embedding_model = None # Set to None if loading fails

# AI Generated Content Detector Model
ai_detector_model_instance = None
if os.path.exists(AI_DETECTOR_MODEL_PATH):
    try:
        ai_detector_model_instance = AIGeneratedDetector(num_classes=2)
        ai_detector_model_instance.load_state_dict(torch.load(AI_DETECTOR_MODEL_PATH, map_location=torch.device('cpu')))
        ai_detector_model_instance.eval() # Set to evaluation mode
        print(f"Loaded AI Detector model from: {AI_DETECTOR_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading AI Detector model from {AI_DETECTOR_MODEL_PATH}: {e}")
        ai_detector_model_instance = None
else:
    print(f"AI Detector model not found at {AI_DETECTOR_MODEL_PATH}. AI detection will be skipped.")


# --- Feature Extraction Functions ---

def get_text_embedding(text):
    if text_tokenizer is None or text_model is None:
        print("Text embedding model not loaded, returning zeros.")
        return np.zeros(TEXT_EMBEDDING_DIM, dtype=np.float32)
    
    # Ensure text is string, handle potential NaN
    if not isinstance(text, str):
        text = str(text)

    inputs = text_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = text_model(**inputs)
    # Use mean of last hidden states as embedding
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding.astype(np.float32)

def get_image_embedding(image_path):
    if image_embedding_model is None or not os.path.exists(image_path):
        print(f"Image embedding model not loaded or image not found at {image_path}, returning zeros.")
        return np.zeros(IMAGE_EMBEDDING_DIM, dtype=np.float32)
    
    try:
        image = Image.open(image_path).convert('RGB')
        # Apply only the transforms needed for feature extraction, then pass to model
        # The model itself has the resizing and normalization built into the torchvision.transforms.Compose
        # For a ResNet feature extractor, the transforms are usually applied before passing to the model
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image_tensor = preprocess(image).unsqueeze(0) # Add batch dimension

        with torch.no_grad():
            embedding = image_embedding_model(image_tensor).squeeze().numpy()
        return embedding.astype(np.float32)
    except Exception as e:
        print(f"Error processing image {image_path} for embedding: {e}, returning zeros.")
        return np.zeros(IMAGE_EMBEDDING_DIM, dtype=np.float32)

def get_ai_detection_score(image_path):
    """
    Returns the probability that an image is AI-generated (class 1).
    Assumes `AIGeneratedDetector` outputs logits for 2 classes (0: Real, 1: Fake/AI).
    """
    if ai_detector_model_instance is None or not os.path.exists(image_path):
        # If model not loaded or image missing, return a default neutral score (e.g., 0.5)
        print(f"AI Detector model not loaded or image not found at {image_path}, returning 0.5 score.")
        return 0.5
    
    try:
        image = Image.open(image_path).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image_tensor = preprocess(image).unsqueeze(0) # Add batch dimension

        with torch.no_grad():
            outputs = ai_detector_model_instance(image_tensor)
            # Apply softmax to get probabilities, then take the probability for the 'fake' class (index 1)
            probabilities = torch.softmax(outputs, dim=1)
            ai_score = probabilities[0, 1].item() # Score for the 'fake' class
        return float(ai_score)
    except Exception as e:
        print(f"Error processing image {image_path} for AI detection: {e}, returning 0.5 score.")
        return 0.5

# Placeholder for video embedding (if you expand to video later)
def get_video_embedding(video_path):
    # This is a placeholder. Real implementation would involve
    # extracting keyframes, getting image embeddings for each, and averaging.
    print(f"Video embedding not implemented for {video_path}, returning zeros.")
    return np.zeros(IMAGE_EMBEDDING_DIM, dtype=np.float32) # Using same dim as image for simplicity

def extract_features(metadata_path, output_path):
    """
    Orchestrates the feature extraction process for all modalities.
    """
    print(f"Loading metadata from: {metadata_path}")
    content_df = pd.read_pickle(metadata_path)

    tqdm.pandas(desc="Extracting text embeddings")
    content_df['text_embedding'] = content_df['text'].progress_apply(get_text_embedding)
    
    tqdm.pandas(desc="Extracting image embeddings")
    # Assuming 'image_path' column points to locally downloaded image files
    content_df['image_embedding'] = content_df['image_path'].progress_apply(get_image_embedding)

    tqdm.pandas(desc="Extracting AI detection scores")
    content_df['ai_image_score'] = content_df['image_path'].progress_apply(get_ai_detection_score)
    
    # Placeholder for video embeddings if needed
    # content_df['video_embedding'] = content_df['video_path'].progress_apply(get_video_embedding)

    # Save the dataframe with all embeddings
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    content_df.to_pickle(output_path)
    print(f"Content embeddings and AI scores saved to: {output_path}")
    print(f"Final DataFrame columns: {content_df.columns.tolist()}")
    return content_df

if __name__ == "__main__":
    print("--- Running Feature Extraction Locally ---")
    
    # Adjusting project_root calculation for running feature_extraction.py as a script
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    import sys
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Now import from config (make sure these paths are defined in src/config.py)
    from src.config import (
        PROCESSED_DATA_DIR, INTERIM_DATA_DIR,
        CONTENT_METADATA_PKL, CONTENT_EMBEDDINGS_PKL,
        AI_DETECTOR_MODEL_PATH # Ensure this is also imported here
    )

    # This dummy setup will work with the simplified content_metadata.pkl
    # that has only 'post_1', 'post_2', 'post_3' and dummy text/image paths
    # (assuming you ran data_ingestion.py and it created dummy image files)

    # Ensure content_metadata.pkl exists from data_ingestion step
    if not os.path.exists(CONTENT_METADATA_PKL):
        print(f"Error: {CONTENT_METADATA_PKL} not found. Please run data_ingestion first.")
    else:
        extract_features(
            metadata_path=CONTENT_METADATA_PKL,
            output_path=CONTENT_EMBEDDINGS_PKL
        )
    print("\nFeature extraction complete.")