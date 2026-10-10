import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# Emotion Model Configuration
EMOTION_DATASET_PATH = DATA_DIR / "raw" / "fer2013" # Adjust if using a CSV vs Image Folders
EMOTION_MODEL_PATH = MODEL_DIR / "emotion_model.keras"
IMG_SHAPE = (48, 48, 3) # VGG19 requires 3 channels
BATCH_SIZE = 64
EPOCHS = 30
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Music Recommendation Configuration
MUSIC_DATASET_PATH = DATA_DIR / "raw" / "spotify_tracks.csv"
PROCESSED_MUSIC_PATH = DATA_DIR / "processed" / "music_features.csv"
RECOMMENDER_MODEL_PATH = MODEL_DIR / "recommender.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# Emotion to Audio Feature Mapping (Target baselines)
EMOTION_PROFILE_MAPPING = {
    'happy':    {'energy': 0.80, 'valence': 0.85, 'danceability': 0.75, 'tempo': 120},
    'sad':      {'energy': 0.35, 'valence': 0.20, 'danceability': 0.40, 'tempo': 75},
    'angry':    {'energy': 0.90, 'valence': 0.25, 'danceability': 0.60, 'tempo': 140},
    'fear':     {'energy': 0.60, 'valence': 0.30, 'danceability': 0.50, 'tempo': 100},
    'surprise': {'energy': 0.85, 'valence': 0.70, 'danceability': 0.65, 'tempo': 130},
    'neutral':  {'energy': 0.50, 'valence': 0.50, 'danceability': 0.50, 'tempo': 100},
    'disgust':  {'energy': 0.40, 'valence': 0.20, 'danceability': 0.40, 'tempo': 80},
}