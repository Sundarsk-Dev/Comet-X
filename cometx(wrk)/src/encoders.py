# FILE: src/encoders.py
import torch
from transformers import AutoTokenizer, AutoModel, ViTFeatureExtractor, ViTModel
from PIL import Image
import requests

# This is a flag to avoid repeated warnings in a development environment
warnings_issued = False

def get_text_embedding(text_content, model_name='roberta-large'):
    """
    Generates a fixed-size embedding vector for a given text string.
    """
    global warnings_issued
    if not text_content or not isinstance(text_content, str):
        print("Invalid input: text_content must be a non-empty string.")
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        if not warnings_issued:
            print("RoBERTa model weights were not initialized from a down-stream task checkpoint.")
            warnings_issued = True

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
    Generates a fixed-size embedding vector for an image from a URL.
    """
    global warnings_issued
    if not image_url or not isinstance(image_url, str):
        print("Invalid input: image_url must be a non-empty string.")
        return None

    try:
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        model = ViTModel.from_pretrained('google/vit-base-patch16-224')

        if not warnings_issued:
            print("ViT model weights were not initialized from a down-stream task checkpoint.")
            warnings_issued = True

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