import torch
from transformers import AutoTokenizer, AutoModel, ViTFeatureExtractor, ViTModel, ViTForImageClassification, ViTImageProcessor
from PIL import Image
import requests
import os
import warnings

# These flags prevent repeated warnings from Hugging Face during execution.
_warnings_issued_roberta = False
_warnings_issued_vit_embedding = False
_warnings_issued_vit_detector = False # New flag for the detector model

def get_text_embedding(text_content, model_name='roberta-large'):
    """
    Generates a fixed-size embedding vector for a given text string using a
    pre-trained language model (RoBERTa-large). This embedding is used for
    multimodal fusion in the GNN.

    Args:
        text_content (str): The raw text string to be encoded.
        model_name (str): The name of the pre-trained model to use.

    Returns:
        torch.Tensor: A 1D tensor representing the text embedding.
                      Returns None if the input is invalid or an error occurs.
    """
    global _warnings_issued_roberta
    if not text_content or not isinstance(text_content, str):
        print("Invalid input: text_content must be a non-empty string.")
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        if not _warnings_issued_roberta:
            print("RoBERTa model weights were not initialized from a down-stream task checkpoint.")
            warnings.warn("You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.", UserWarning)
            _warnings_issued_roberta = True

        inputs = tokenizer(
            text_content,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)

        text_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        return text_embedding

    except Exception as e:
        print(f"An error occurred during text embedding generation: {e}")
        return None

def get_image_embedding(image_url):
    """
    Generates a fixed-size embedding vector for an image from a URL using a
    pre-trained Vision Transformer (ViT) model. This embedding is specifically
    used for multimodal fusion as a feature for the GNN.

    Args:
        image_url (str): The URL of the image to be encoded.

    Returns:
        torch.Tensor: A 1D tensor representing the image embedding.
                      Returns None if the image cannot be processed.
    """
    global _warnings_issued_vit_embedding
    if not image_url or not isinstance(image_url, str):
        print("Invalid input: image_url must be a non-empty string.")
        return None

    try:
        # This model is used for general image feature extraction for the GNN
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        model = ViTModel.from_pretrained('google/vit-base-patch16-224')

        if not _warnings_issued_vit_embedding:
            print("ViT (embedding) model weights were not initialized from a down-stream task checkpoint.")
            warnings.warn("You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.", UserWarning)
            _warnings_issued_vit_embedding = True

        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw).convert('RGB')

        inputs = feature_extractor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        image_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        return image_embedding

    except Exception as e:
        print(f"An error occurred during image embedding generation: {e}")
        return None

def detect_ai_generated_image(image_url, model_path="models/vit_cifake_detector_final"): # <-- CORRECTED PATH HERE
    """
    Loads your fine-tuned ViT model (vit_cifake_detector_final) and predicts
    if an image is AI-generated or real. This is your custom AI detection.
    
    Args:
        image_url (str): The URL of the image to check.
        model_path (str): The local path to your fine-tuned model directory.
        
    Returns:
        dict: A dictionary with 'predicted_label' (str) and 'confidence' (float),
              or None if an error occurs.
              Labels are 'AI-Generated' or 'Real'.
    """
    global _warnings_issued_vit_detector
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_path):
        print(f"Error: AI Image Detector model not found at '{model_path}'.")
        print("Please ensure the 'vit_cifake_detector_final' folder is correctly placed in the 'models/' directory relative to where main.py is run.")
        return None

    try:
        # Load your fine-tuned model and its processor
        model = ViTForImageClassification.from_pretrained(model_path).to(device)
        image_processor = ViTImageProcessor.from_pretrained(model_path)

        # Suppress the "Some weights were not initialized" warning for this specific model load
        # as it's expected for a fine-tuned model.
        if not _warnings_issued_vit_detector:
            print(f"Loading custom AI Image Detector from {model_path}. Expecting fine-tuned weights.")
            _warnings_issued_vit_detector = True

        # Download image from URL
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")

        model.eval() # Set model to evaluation mode for inference

        inputs = image_processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            
            confidence, predicted_class_id = torch.max(probabilities, dim=1)
            
            # Retrieve the label name from the model's configuration.
            # Assuming your model's config.json has id2label mapping.
            # For CIFake, 0 typically maps to 'fake' and 1 to 'real'.
            # Verify model.config.id2label to confirm your specific mapping.
            predicted_label_raw = model.config.id2label[predicted_class_id.item()]
            
            # Map to a more user-friendly label
            if predicted_label_raw.lower() == 'fake':
                final_label = "AI-Generated"
            elif predicted_label_raw.lower() == 'real':
                final_label = "Real(but does not prove whether it is fake news)"
            else: # Fallback for unexpected labels
                final_label = predicted_label_raw


        return {
            "predicted_label": final_label,
            "confidence": confidence.item()
        }

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image for AI detection: {e}")
        return None
    except Exception as e:
        print(f"An error occurred during AI image detection: {e}")
        return None
