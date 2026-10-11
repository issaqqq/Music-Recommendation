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

        # Load OpenCV's pre-trained Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def preprocess_face(self, face_crop):
        """Resizes and normalizes the cropped face for the model."""
        # Resize to 48x48
        face_resized = cv2.resize(face_crop, (IMG_SHAPE[0], IMG_SHAPE[1]))
        
        # Convert grayscale to 3 channels (VGG19 expects RGB)
        if len(face_resized.shape) == 2:
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
        else:
            face_rgb = face_resized

        # Normalize pixel values (same as ImageDataGenerator rescale=1./255)
        face_normalized = face_rgb.astype('float32') / 255.0
        
        # Expand dimensions to match batch shape (1, 48, 48, 3)
        face_expanded = np.expand_dims(face_normalized, axis=0)
        
        return face_expanded

    def detect_and_predict(self, frame):
        """Detects face in a frame and predicts emotion."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.3, 
            minNeighbors=5, 
            minSize=(30, 30)
        )

        results = []

        for (x, y, w, h) in faces:
            # Crop the face region
            face_crop = gray_frame[y:y+h, x:x+w]
            
            # Preprocess
            processed_face = self.preprocess_face(face_crop)
            
            # Predict
            preds = self.model.predict(processed_face, verbose=0)[0]
            max_index = np.argmax(preds)
            
            emotion = EMOTIONS[max_index]
            confidence = preds[max_index]
            
            # Store probabilities for all classes
            probabilities = {EMOTIONS[i]: float(preds[i]) for i in range(len(EMOTIONS))}
            
            results.append({
                'box': (x, y, w, h),
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': probabilities
            })

        return results

def test_webcam():
    """Standalone function to test the webcam and model."""
    predictor = EmotionPredictor()
    cap = cv2.VideoCapture(0) # 0 is usually the default laptop webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Get predictions
        results = predictor.detect_and_predict(frame)

        # Draw bounding boxes and text
        for result in results:
            x, y, w, h = result['box']
            emotion = result['emotion']
            conf = result['confidence'] * 100

            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw text
            text = f"{emotion.capitalize()} ({conf:.1f}%)"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Emotion Detection Test', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_webcam()