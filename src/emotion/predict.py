import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
from pathlib import Path

# Add project root to sys.path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import EMOTION_MODEL_PATH, IMG_SHAPE, EMOTIONS

class EmotionPredictor:
    def __init__(self):
        print("Loading Emotion Model...")
        try:
            self.model = load_model(str(EMOTION_MODEL_PATH))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

        # Initialize Haar Cascade safely
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise IOError("Failed to load cascade XML file.")
        except Exception as e:
            print(f"Warning: Haar Cascade initialization failed ({e}). Trying alternative detection setup...")
            self.face_cascade = None

    def preprocess_face(self, face_crop):
        """Resizes, converts to 1-channel grayscale, and normalizes."""
        # Resize to the expected dimension (48x48) using config base dimensions
        face_resized = cv2.resize(face_crop, (IMG_SHAPE[0], IMG_SHAPE[1]))
        
        # If image is 3 channels (colored), convert it to single channel (grayscale)
        if len(face_resized.shape) == 3:
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

        # Normalize pixels to values between 0 and 1
        face_normalized = face_resized.astype('float32') / 255.0
        
        # Expand dimensions to match model input shape: (1, 48, 48, 1)
        face_expanded = np.expand_dims(face_normalized, axis=(0, -1))
        
        return face_expanded

    def detect_and_predict(self, frame):
        """Detects face in a frame and predicts emotion."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = []

        if self.face_cascade is not None:
            faces = self.face_cascade.detectMultiScale(
                gray_frame, 
                scaleFactor=1.3, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
        else:
            # Fallback: Treat the center crop of the frame as the face if cascade fails
            h_f, w_f, _ = frame.shape
            faces = [[w_f // 4, h_f // 4, w_f // 2, h_f // 2]]

        for (x, y, w, h) in faces:
            face_crop = gray_frame[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue
            
            processed_face = self.preprocess_face(face_crop)
            preds = self.model.predict(processed_face, verbose=0)[0]
            max_index = np.argmax(preds)
            
            emotion = EMOTIONS[max_index]
            confidence = preds[max_index]
            probabilities = {EMOTIONS[i]: float(preds[i]) for i in range(len(EMOTIONS))}
            
            results.append({
                'box': (x, y, w, h),
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': probabilities
            })

        return results

def test_webcam():
    predictor = EmotionPredictor()
    
    # Auto-detect camera index 
    cap = None
    for idx in range(3):
        print(f"Trying camera index {idx}...")
        temp_cap = cv2.VideoCapture(idx)
        if temp_cap.isOpened():
            ret, test_frame = temp_cap.read()
            if ret:
                cap = temp_cap
                print(f"Successfully connected to camera index {idx}")
                break
            temp_cap.release()

    if cap is None:
        print("Error: Could not open any webcam or stream index (0, 1, 2).")
        return

    print("Webcam stream active. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        
        # Try model prediction
        try:
            results = predictor.detect_and_predict(frame)
        except Exception:
            results = []

        for result in results:
            x, y, w, h = result['box']
            emotion = result['emotion']
            conf = result['confidence'] * 100

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{emotion.capitalize()} ({conf:.1f}%)"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Emotion Detection Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_webcam()