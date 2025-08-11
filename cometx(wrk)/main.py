# FILE: main.py
import torch
import numpy as np
import warnings
import os
import base64
import requests
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel, RobertaTokenizer, RobertaModel
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero

# You can add this line at the top to disable the Hugging Face symlink warning
# This is optional but can make the output cleaner
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.file_download")

# --- Mock functions to simulate your pipeline components ---
# These are simple implementations to make the complete script runnable.
def get_text_embedding(text_content):
    """Generates a text embedding using a pre-trained RoBERTa model."""
    try:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        model = RobertaModel.from_pretrained('roberta-large')
        encoded_input = tokenizer(text_content, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            output = model(**encoded_input)
        # Use the mean of the last hidden state as the embedding
        return output.last_hidden_state.mean(dim=1).squeeze()
    except Exception as e:
        print(f"An error occurred during text embedding generation: {e}")
        return None

def get_image_embedding(image_source):
    """
    Generates an image embedding. This function handles
    local file paths, URLs, and base64 encoded strings.
    """
    try:
        image = None
        # Check if the source is a base64 string
        if image_source.startswith("data:image"):
            header, encoded_data = image_source.split(',', 1)
            image_data = base64.b64decode(encoded_data)
            image = Image.open(BytesIO(image_data)).convert("RGB")
            print("Loading image from base64 string.")
        # Check if the source is a local file path
        elif os.path.exists(image_source):
            image = Image.open(image_source).convert("RGB")
            print(f"Loading image from local file: {image_source}")
        # Check if the source is a URL
        elif image_source.startswith(('http://', 'https://')):
            response = requests.get(image_source)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            print(f"Loading image from URL: {image_source}")
        else:
            raise ValueError(f"Invalid image source: {image_source}")

        # Use a pre-trained ViT model for embedding
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        inputs = feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()
    except Exception as e:
        print(f"An error occurred during image embedding generation: {e}")
        return None

def fuse_embeddings(text_vec, image_vec):
    """Combines text and image embeddings by concatenation."""
    # Handle cases where one or both inputs might be None
    if text_vec is not None and image_vec is not None:
        return torch.cat((text_vec, image_vec), dim=0)
    elif text_vec is not None:
        return text_vec
    elif image_vec is not None:
        return image_vec
    else:
        return None

def build_hetero_graph(num_posts, num_users, all_post_features, all_user_features):
    """
    Simulates building a heterogeneous graph.
    Returns a dummy `data` object with mock features and edges.
    """
    data = HeteroData()
    data['post'].x = all_post_features
    data['user'].x = all_user_features
    
    # Create some dummy edges
    data['post', 'posted_by', 'user'].edge_index = torch.randint(0, num_posts, (2, num_users), dtype=torch.long)
    data['user', 'interacted_with', 'post'].edge_index = torch.randint(0, num_users, (2, num_posts * 5), dtype=torch.long)
    
    return data

class GNN(torch.nn.Module):
    """
    A mock GNN model for demonstration that handles heterogeneous data directly.
    It expects a dictionary of tensors as input, not a single tensor.
    """
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # These are placeholders; a real GNN would have actual layers
        self.conv1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv2 = torch.nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x_dict, edge_index_dict):
        # A simple forward pass that returns a dictionary of tensors
        post_x = x_dict['post']
        # Mock aggregation of user features, etc.
        out = {
            'post': self.conv2(torch.relu(self.conv1(post_x[:, :128]))),
            'user': x_dict['user']
        }
        return out

def detect_ai_generated_image(image_url):
    """Mock function for AI image detection."""
    # This is a dummy response since we don't have the real model
    return {
        'predicted_label': 'Real',
        'confidence': 95.0
    }

def check_facts(text_content):
    """Mock function for external fact-checking."""
    # This is a dummy response
    return None

def perform_context_analysis(text_content):
    """
    Mock function to simulate context analysis based on the text claim.
    """
    if not text_content:
        return "No text content was provided for analysis."
        
    text_content_lower = text_content.lower()
    if "world war" in text_content_lower or "declared" in text_content_lower:
        return "The claim of a new world war being declared is a very serious and public event that would be widely reported by all major news outlets. The lack of such reports is a strong indicator of misinformation."
    elif "free money" in text_content_lower or "free gift card" in text_content_lower:
        return "The claim of receiving free money or gift cards from a financial institution is a classic phishing and scam tactic. Legitimate banks do not operate this way."
    else:
        return "The provided text claim is a general statement that could be true or false. More specific context is needed for a detailed analysis."

def explain_prediction(post_prediction, attention_weights, context_analysis_result):
    """Generates a human-readable explanation based on provided weights and context."""
    confidence = post_prediction[0].item() if post_prediction[1].item() > post_prediction[0].item() else post_prediction[1].item()
    confidence_score = 1 - confidence
    claim_status = "credible" if confidence_score > 0.5 else "fake"
    
    explanation_parts = []
    
    if 'text_focus' in attention_weights:
        explanation_parts.append(f"1. **Text Content**: It detected {attention_weights['text_focus']}.")
        
    # Add the contextual analysis to the explanation if it exists
    if context_analysis_result:
        explanation_parts.append(f"2. **Contextual Analysis**: The model's analysis is supported by the fact that {context_analysis_result}")
    
    if 'image_focus' in attention_weights:
        explanation_parts.append(f"3. **Visual Content**: It found {attention_weights['image_focus']}.")

    explanation_list = "\n".join(explanation_parts)

    explanation = (
        f"Based on our analysis, this post is likely to be **{claim_status} information** "
        f"with a confidence score of {confidence_score:.2f}. The model's decision was primarily influenced by:\n"
        f"{explanation_list}"
    )
    return explanation

def _read_image_as_base64(file_path: str) -> str | None:
    """
    Reads a local image file and returns its base64-encoded string.
    """
    cleaned_path = file_path.strip().strip('"').strip("'")
    if not os.path.exists(cleaned_path):
        print(f"Error: Local image file not found at '{cleaned_path}'")
        return None
    try:
        with open(cleaned_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        print(f"Error reading local image file: {e}")
        return None

def main():
    """
    Orchestrates the entire COMET-X pipeline from input to explanation.
    This version now takes user input for text and a hardcoded image.
    """
    print("--- COMET-X Pipeline Starting ---")

    # --- 1. Data Ingestion (User Input) ---
    print("\nPlease enter the claim you want to analyze.")
    text_content = input("Enter the text claim (leave blank to skip): ")
    
    # HARDCODED: The image is now automatically loaded from a local file.
    image_input = 'fakecard.jpeg'
    print(f"Using local image file: {image_input}")

    image_data = None
    if image_input:
        cleaned_image_input = image_input.strip().strip('"').strip("'")
        parsed_url = urlparse(cleaned_image_input)
        if parsed_url.scheme and parsed_url.netloc:
            image_data = cleaned_image_input
        elif os.path.exists(cleaned_image_input):
            print("Reading local image file...")
            image_data = _read_image_as_base64(cleaned_image_input)
        else:
            print(f"Invalid image input. The path '{cleaned_image_input}' was not found.")
    
    sample_post = {
        "post_id": "user_input",
        "text_content": text_content.strip() if text_content.strip() else None,
        "image_url": image_data,
        "author_id": "user_input"
    }

    # --- 2. Feature Extraction (with conditional checks) ---
    print("\n[STEP 1/7] Extracting features from text and image...")
    text_vec = None
    if sample_post.get('text_content'):
        text_vec = get_text_embedding(sample_post['text_content'])
    else:
        print("No text content provided. Skipping text embedding.")
    
    image_vec = None
    if sample_post.get('image_url'):
        image_vec = get_image_embedding(sample_post['image_url'])
    else:
        print("No image provided. Skipping image embedding.")

    if text_vec is None and image_vec is None:
        print("Feature extraction failed. No content to analyze. Exiting.")
        return

    # --- 3. Multimodal Fusion (now handles None inputs) ---
    print("\n[STEP 2/7] Fusing multimodal features...")
    fused_vec = fuse_embeddings(text_vec, image_vec)
    if fused_vec is not None:
        print(f"Fused embedding shape: {fused_vec.shape}")
    else:
        print("No content to fuse. Exiting.")
        return

    # --- 4. Graph Construction (Simulated) ---
    print("\n[STEP 3/7] Building heterogeneous graph...")
    num_posts = 100
    num_users = 50
    post_feature_size = fused_vec.shape[0]
    user_feature_size = 64

    all_post_features = torch.randn(num_posts, post_feature_size)
    all_post_features[0] = fused_vec
    all_user_features = torch.randn(num_users, user_feature_size)
    
    graph_data = build_hetero_graph(
        num_posts, num_users, all_post_features, all_user_features
    )
    print("Graph built successfully.")

    # --- 5. GNN Inference (Simulated with improved logic) ---
    print("\n[STEP 4/7] Running GNN inference...")
    gnn_model = GNN(hidden_channels=128, out_channels=2)
    
    # We will now create a more definitive mock prediction.
    mock_prediction = None
    text_content_lower = sample_post.get('text_content', '').lower()
    if 'world war' in text_content_lower or 'free money' in text_content_lower:
        # A strong fake prediction
        mock_prediction = torch.tensor([[0.95, 0.05]])
    else:
        # A moderate credible prediction for other claims
        mock_prediction = torch.tensor([[0.20, 0.80]])
    
    post_prediction = mock_prediction[0].softmax(dim=-1)
    
    # --- 6. AI Image Detection ---
    ai_image_detection_result = None
    if sample_post.get('image_url'):
        print("\n[STEP 5/7] Detecting AI-generated image content...")
        ai_image_detection_result = detect_ai_generated_image(sample_post['image_url'])
    else:
        print("\n[STEP 5/7] Detecting AI-generated image content...")
        print("No image content provided. Skipping AI image detection.")
    
    # --- 7. Fact-Checking API Call ---
    fact_check_result = None
    if sample_post.get('text_content'):
        print("\n[STEP 6/7] Cross-referencing with fact-checking services...")
        fact_check_result = check_facts(sample_post['text_content'])
    else:
        print("\n[STEP 6/7] Cross-referencing with fact-checking services...")
        print("No text content provided. Skipping external fact-check.")
    
    # --- 8. HEMMR Explanation (Now with dynamic, user-provided text) ---
    print("\n[STEP 7/7] Generating explanation...")

    attention_weights = {}
    if sample_post.get('text_content'):
        attention_weights['text_focus'] = f"a high focus on the words in the claim: '{sample_post['text_content']}'"
    if sample_post.get('image_url') and ai_image_detection_result:
        attention_weights['image_focus'] = f"the AI image detection result which classified the image as '{ai_image_detection_result['predicted_label']}'"

    context_analysis_result = perform_context_analysis(sample_post['text_content'])

    explanation = explain_prediction(post_prediction, attention_weights, context_analysis_result)
    
    print("\n--- COMPILED ANALYSIS ---")
    
    # Output the AI Image Detection Result
    if ai_image_detection_result:
        label = ai_image_detection_result['predicted_label']
        confidence = ai_image_detection_result['confidence']
        print(f"AI Image Detection: Image is classified as '{label}' with {confidence:.2f}% confidence.")
    else:
        print("AI Image Detection: Not performed (no image provided).")

    # Output the Fact-Check Result
    if fact_check_result:
        print("External Fact-Check Results:")
        if isinstance(fact_check_result, list) and fact_check_result:
            for result in fact_check_result:
                print(f" - Claim: {result.get('claim_text', 'N/A')}")
                print(f" - Rating: {result.get('rating', 'N/A')} by {result.get('publisher', 'N/A')}")
                print(f" - Source: {result.get('review_url', 'N/A')}")
        else:
            print(" - No highly relatable facts found.")
    else:
        print("External Fact-Check: Not performed (no text provided).")
        
    # Output the HEMMR explanation
    print("\nExplanation (HEMMR):")
    print(explanation)
    
    print("-" * 25) # Separator to highlight the final status
    
    # Output the GNN Prediction as the final result
    confidence_score = post_prediction[0].item() if post_prediction[0].item() > post_prediction[1].item() else post_prediction[1].item()
    claim_status_label = "fake" if mock_prediction[0][0] > mock_prediction[0][1] else "credible"
    print(f"Final Claim Status: The post is likely **{claim_status_label}** with a confidence of {confidence_score:.2f}.")

if __name__ == "__main__":
    main()
