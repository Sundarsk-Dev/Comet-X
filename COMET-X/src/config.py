# src/config.py
import os

# Root directory of the project (assuming this file is in src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define main directories
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Define sub-directories within data
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
INTERIM_DATA_DIR = os.path.join(DATA_DIR, 'interim')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Ensure all necessary directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
# Create sub-directory for AI detector models
os.makedirs(os.path.join(MODELS_DIR, 'ai_detectors'), exist_ok=True)
# Create sub-directory for GNN models
os.makedirs(os.path.join(MODELS_DIR, 'gnn_models'), exist_ok=True)


# --- Dataset-specific configurations ---
FAKEDDIT_SUBSET_CSV = os.path.join(RAW_DATA_DIR, 'fakeddit_subset.csv')
CIFAKE_IMAGES_DIR = os.path.join(RAW_DATA_DIR, 'CIFake_Dataset') # Assuming this structure for CIFAKE

# --- Model names/paths (for pre-trained or saved models) ---
TEXT_EMBEDDING_MODEL = 'distilbert-base-uncased' # Renamed for clarity with transformers library
IMAGE_EMBEDDING_MODEL = 'resnet18' # For feature extraction backbone

# Updated: Specify the actual trained model file name
AI_DETECTOR_MODEL_PATH = os.path.join(MODELS_DIR, 'ai_detectors', 'trained_ai_detector.pth')
GNN_MODEL_PATH = os.path.join(MODELS_DIR, 'gnn_models', 'trained_gnn.pth') # Path for saved GNN model


# --- Output file paths for processed data stages ---
PROCESSED_METADATA_PKL = os.path.join(INTERIM_DATA_DIR, 'processed_metadata.pkl') # Changed to PKL as it's a DataFrame
CONTENT_EMBEDDINGS_PKL = os.path.join(PROCESSED_DATA_DIR, 'content_embeddings.pkl')
GRAPH_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'multimodal_graph.pt') # Updated name for consistency

# --- Hyperparameters (Example - expand as needed) ---
# Feature extraction dimensions
TEXT_EMBEDDING_DIM = 768 # For DistilBERT-base-uncased
IMAGE_EMBEDDING_DIM = 512 # For ResNet18 features (before final classification layer)
# If you used a different model for image embeddings, this might change.

# GNN training
GNN_HIDDEN_DIM = 128
GNN_OUTPUT_DIM = 2 # For binary classification (fake/real)
GNN_LEARNING_RATE = 0.001
GNN_EPOCHS = 50

# AI Detector training
AI_DETECTOR_LEARNING_RATE = 0.001
AI_DETECTOR_EPOCHS = 10
AI_DETECTOR_BATCH_SIZE = 32

# Combined feature dimension for content nodes in GNN
# TEXT_EMBEDDING_DIM (768) + IMAGE_EMBEDDING_DIM (512) + AI_IMAGE_SCORE (1)
GNN_INPUT_DIM_CONTENT = TEXT_EMBEDDING_DIM + IMAGE_EMBEDDING_DIM + 1
# User node feature dimension (from your dummy data)
GNN_INPUT_DIM_USER = 1