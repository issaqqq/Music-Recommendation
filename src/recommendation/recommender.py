import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pickle
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import MUSIC_DATASET_PATH, PROCESSED_MUSIC_PATH, RECOMMENDER_MODEL_PATH, SCALER_PATH, EMOTION_PROFILE_MAPPING

class MusicRecommender:
    def __init__(self):
        self.feature_cols = ['energy', 'valence', 'danceability', 'tempo']
        self.df = None
        self.scaler = None
        self.model = None

    def prepare_dataset(self):
        """Loads and processes the raw Spotify CSV dataset with flexible column mapping."""
        if not os.path.exists(MUSIC_DATASET_PATH):
            print(f"Error: Music dataset not found at {MUSIC_DATASET_PATH}")
            print("Please download a Spotify tracks CSV dataset from Kaggle and save it as 'spotify_tracks.csv' in data/raw/")
            return False

        print("Processing music dataset...")
        df = pd.read_csv(MUSIC_DATASET_PATH)
        
        # Clean column names (strip spaces, lowercase)
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Flexible mapping for track name
        if 'track_name' not in df.columns:
            for alt in ['track', 'name', 'song_name', 'song']:
                if alt in df.columns:
                    df['track_name'] = df[alt]
                    break

        # Flexible mapping for artist name
        if 'artist_name' not in df.columns:
            for alt in ['artist', 'artists', 'artist_name', 'singer']:
                if alt in df.columns:
                    df['artist_name'] = df[alt]
                    break

        # Verify required columns
        required = ['track_name', 'artist_name'] + self.feature_cols
        for col in required:
            if col not in df.columns:
                print(f"Missing required column in dataset: '{col}'. Found columns: {list(df.columns)}")
                return False

        # Drop missing rows for features
        df = df.dropna(subset=required)
        
        # Save processed dataset
        os.makedirs(os.path.dirname(PROCESSED_MUSIC_PATH), exist_ok=True)
        df.to_csv(PROCESSED_MUSIC_PATH, index=False)
        self.df = df
        print(f"Dataset processed successfully with {len(df)} tracks.")
        return True

    def train_recommender(self):
        """Fits a Standard Scaler and KNN model on the song features."""
        if self.df is None:
            if os.path.exists(PROCESSED_MUSIC_PATH):
                self.df = pd.read_csv(PROCESSED_MUSIC_PATH)
            else:
                success = self.prepare_dataset()
                if not success:
                    return False

        print("Training KNN Recommendation Engine...")
        X = self.df[self.feature_cols].values

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Fit Nearest Neighbors using cosine distance
        self.model = NearestNeighbors(n_neighbors=20, metric='cosine', algorithm='brute')
        self.model.fit(X_scaled)

        # Save model and scaler
        os.makedirs(os.path.dirname(RECOMMENDER_MODEL_PATH), exist_ok=True)
        with open(RECOMMENDER_MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print("Recommender model and scaler saved successfully.")
        return True

    def load_model(self):
        """Loads trained KNN model and scaler."""
        if os.path.exists(RECOMMENDER_MODEL_PATH) and os.path.exists(SCALER_PATH):
            with open(RECOMMENDER_MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
            if os.path.exists(PROCESSED_MUSIC_PATH):
                self.df = pd.read_csv(PROCESSED_MUSIC_PATH)
            return True
        return False

    def recommend_for_emotion(self, emotion, top_n=10, genre_filter=None):
        """Returns top N song recommendations based on emotion profile with diversity filtering."""
        if self.model is None or self.scaler is None:
            loaded = self.load_model()
            if not loaded:
                print("Recommender model not found. Training now...")
                success = self.train_recommender()
                if not success:
                    return []

        # Get target profile features for the emotion
        profile = EMOTION_PROFILE_MAPPING.get(emotion, EMOTION_PROFILE_MAPPING['neutral'])
        query_vector = np.array([[
            profile['energy'],
            profile['valence'],
            profile['danceability'],
            profile['tempo']
        ]])

        # Scale query vector
        query_scaled = self.scaler.transform(query_vector)

        # Find nearest neighbors
        fetch_k = top_n * 3
        distances, indices = self.model.kneighbors(query_scaled, n_neighbors=min(fetch_k, len(self.df)))

        recommendations = []
        for idx in indices[0]:
            row = self.df.iloc[idx]
            
            if genre_filter and genre_filter != "All":
                if 'genre' in row and genre_filter.lower() not in str(row['genre']).lower():
                    continue

            artist = row['artist_name']
            
            # Artist diversity logic
            artist_count = sum(1 for r in recommendations if r['artist_name'] == artist)
            if artist_count >= 1 and len(recommendations) < top_n - 2:
                continue

            recommendations.append({
                'track_name': row['track_name'],
                'artist_name': artist,
                'genre': row.get('genre', 'Various'),
                'energy': row['energy'],
                'valence': row['valence'],
                'danceability': row['danceability']
            })

            if len(recommendations) >= top_n:
                break

        return recommendations

if __name__ == "__main__":
    rec = MusicRecommender()
    if rec.train_recommender():
        print("\nTest Recommendation for 'happy':")
        print(rec.recommend_for_emotion('happy', top_n=5))