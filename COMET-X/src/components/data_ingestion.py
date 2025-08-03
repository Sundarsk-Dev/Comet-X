# src/components/data_ingestion.py
import pandas as pd
import os
import requests
from PIL import Image
import cv2
import numpy as np
import re
from tqdm import tqdm # For progress bar

# Import configurations
from src.config import FAKEDDIT_SUBSET_CSV, INTERIM_DATA_DIR, PROCESSED_METADATA_CSV

def clean_text(text):
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) # Remove non-alphanumeric characters
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with single space
    return text

def download_image(url, save_path):
    """Downloads an image from a URL and saves it."""
    if pd.isna(url) or not url.startswith(('http://', 'https://')):
        return None
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes
        with open(save_path, 'wb') as out_file:
            for chunk in response.iter_content(chunk_size=8192):
                out_file.write(chunk)
        return save_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None
    except Exception as e:
        print(f"General error saving image from {url}: {e}")
        return None

def extract_keyframes(video_path, output_dir, frames_per_second=1):
    """
    Extracts keyframes from a video.
    For PoC, this is a simplified version.
    """
    if not video_path or not os.path.exists(video_path):
        return []

    os.makedirs(output_dir, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Warning: Could not get FPS for video {video_path}")
        return []

    frame_interval = int(fps / frames_per_second)
    count = 0
    frame_count = 0
    keyframes_paths = []

    while True:
        success, image = vidcap.read()
        if not success:
            break
        if count % frame_interval == 0:
            frame_name = f"frame_{frame_count:04d}.jpg"
            frame_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(frame_path, image)
            keyframes_paths.append(frame_path)
            frame_count += 1
        count += 1
    vidcap.release()
    return keyframes_paths

# --- Main Ingestion Function ---
def ingest_and_preprocess_data(raw_csv_path, interim_output_csv_path, image_save_dir, video_save_dir):
    """
    Ingests raw multimodal data, cleans text, downloads images/videos,
    and saves preprocessed metadata.
    """
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(video_save_dir, exist_ok=True) # For future video support

    print(f"Loading raw data from: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)

    # Select relevant columns and rename if necessary (adjust based on your Fakeddit CSV)
    # Example: Assuming 'id', 'title', 'image_url', 'label', 'created_utc' columns exist
    required_cols = ['id', 'title', 'image_url', 'label', 'created_utc']
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: Not all required columns {required_cols} found in CSV. Please adjust.")
        # Attempt to proceed with available columns, or raise error.
        # For PoC, let's assume 'id', 'text', 'image_url', 'label' for simplicity
        if 'text' not in df.columns and 'title' in df.columns:
            df['text'] = df['title'] # Use title as text if 'text' column is missing
        elif 'text' not in df.columns:
            df['text'] = "" # Fallback
        if 'image_url' not in df.columns:
            df['image_url'] = None # Fallback

        # Filter for essential columns for this PoC:
        df = df[['id', 'text', 'image_url', 'label']].copy()
        df = df.rename(columns={'id': 'content_id'}) # Standardize content_id
    else:
        df = df[required_cols].rename(columns={'id': 'content_id', 'title': 'text', 'created_utc': 'timestamp'})


    print("Cleaning text...")
    df['cleaned_text'] = df['text'].apply(clean_text)

    print("Downloading images...")
    df['image_path'] = None # Initialize column
    # Use tqdm to show progress for downloads
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Downloading images"):
        content_id = row['content_id']
        image_url = row['image_url']

        if pd.notna(image_url) and image_url.strip() != '': # Ensure URL is not empty/NaN
            # Create a unique filename for the image
            image_filename = f"{content_id}_{os.path.basename(image_url).split('?')[0]}"
            image_save_path = os.path.join(image_save_dir, image_filename)

            if not os.path.exists(image_save_path): # Only download if not already present
                path = download_image(image_url, image_save_path)
                df.loc[index, 'image_path'] = path
            else:
                df.loc[index, 'image_path'] = image_save_path
        else:
            df.loc[index, 'image_path'] = None # Explicitly set to None if no URL

    # For PoC, video extraction can be minimal or skipped if dataset lacks video
    df['video_path'] = None # Placeholder for now
    df['keyframes_paths'] = None # Placeholder for now

    # Convert label to integer (e.g., 0 for real, 1 for fake)
    # Adjust based on your dataset's actual labels (e.g., "real", "fake")
    df['label_encoded'] = df['label'].astype('category').cat.codes

    # Filter for essential output columns
    output_df = df[['content_id', 'cleaned_text', 'image_path', 'video_path', 'keyframes_paths', 'label_encoded']].copy()

    print(f"Saving preprocessed metadata to: {interim_output_csv_path}")
    output_df.to_csv(interim_output_csv_path, index=False)
    print("Data ingestion and preprocessing complete.")
    return output_df

if __name__ == "__main__":
    # Example usage for testing this module
    # Ensure fakeddit_subset.csv is in data/raw/
    # and that it has 'id', 'title'/'text', 'image_url', 'label' columns

    # Create a dummy fakeddit_subset.csv if it doesn't exist for testing
    if not os.path.exists(FAKEDDIT_SUBSET_CSV):
        print(f"Creating a dummy {os.path.basename(FAKEDDIT_SUBSET_CSV)} for testing.")
        dummy_data = {
            'id': ['post_1', 'post_2', 'post_3'],
            'title': ['Breaking News: Sun is now square!', 'Cat enjoys a nap in a box.', 'Aliens landed in my backyard yesterday!'],
            'image_url': [
                'https://via.placeholder.com/150/FF0000/FFFFFF?text=Fake_News_Img',
                'https://via.placeholder.com/150/00FF00/FFFFFF?text=Real_News_Img',
                'https://via.placeholder.com/150/0000FF/FFFFFF?text=Alien_Img'
            ],
            'label': ['fake', 'real', 'fake'],
            'created_utc': [1678886400, 1678886500, 1678886600] # Example timestamps
        }
        dummy_df = pd.DataFrame(dummy_data)
        dummy_df.to_csv(FAKEDDIT_SUBSET_CSV, index=False)
        print("Dummy CSV created. Please replace with actual Fakeddit subset for real data.")

    image_output_dir = os.path.join(INTERIM_DATA_DIR, 'images')
    video_output_dir = os.path.join(INTERIM_DATA_DIR, 'videos') # For future

    ingest_and_preprocess_data(
        raw_csv_path=FAKEDDIT_SUBSET_CSV,
        interim_output_csv_path=PROCESSED_METADATA_CSV,
        image_save_dir=image_output_dir,
        video_save_dir=video_output_dir
    )