# 🎶 Emotion-Based Music Recommendation System

An end-to-end Machine Learning and Computer Vision application that detects a user's facial emotion in real-time and recommends music tailored to their exact mood. 

Built with **TensorFlow/Keras (VGG19)** for deep learning emotion classification, **Scikit-Learn (KNN)** for audio feature-based music recommendation, and **Streamlit** for an interactive web dashboard.

---

## 🧠 How It Works

The system operates in a seamless pipeline:
1. **Face Capture & Detection:** The Streamlit UI captures an image from the user's webcam (or an uploaded photo) and uses OpenCV to detect and isolate the user's face.
2. **Emotion Recognition:** The cropped face is passed through a custom-trained Convolutional Neural Network (using VGG19 Transfer Learning), which predicts the dominant emotion (Happy, Sad, Angry, Fear, Surprise, Neutral, Disgust) and calculates a confidence score.
3. **Emotion-to-Audio Mapping:** The predicted emotion is mapped to a target "audio feature profile" (e.g., Happy = high energy, high valence, high tempo).
4. **Music Recommendation:** A K-Nearest Neighbors (KNN) algorithm calculates the cosine similarity between the target emotion profile and a dataset of over 110,000 Spotify tracks, returning the best matching songs.
5. **Feedback Loop:** Users can Like, Dislike, or Skip recommendations, storing interaction data for future personalization.

## 🛠️ Technology Stack
* **Language:** Python 3.10 / 3.11
* **Deep Learning:** TensorFlow, Keras (VGG19)
* **Computer Vision:** OpenCV
* **Machine Learning:** Scikit-learn, Pandas, NumPy
* **Frontend:** Streamlit

---

## 🚀 Steps to Recreate on Your Local PC

Follow these steps to set up, train, and run the project from scratch.

### Step 1: Project Setup
1. Clone this repository or create the project folder structure on your PC.
2. It is highly recommended to use **Python 3.10 or 3.11** (newer versions like 3.14 may lack pre-built OpenCV wheels).
3. Open your terminal and install the required dependencies:
   ```bash
   pip install tensorflow opencv-python scikit-learn pandas numpy matplotlib seaborn streamlit
   ```

### Step 2: Download the Datasets
You need two raw datasets from Kaggle to train the models.
1. **FER-2013 (Emotion Dataset):**
   * Download the organized image folder version of FER-2013 from Kaggle.
   * Extract it and place the `train` and `test` folders inside: `data/raw/fer2013/`
2. **Spotify Tracks Dataset:**
   * Download a Spotify tracks audio features CSV from Kaggle.
   * Rename the file to `spotify_tracks.csv` and place it here: `data/raw/spotify_tracks.csv`

Your directory structure should look exactly like this:
```text
emotion-music-recommender/
├── data/
│   └── raw/
│       ├── fer2013/
│       │   ├── train/
│       │   └── test/
│       └── spotify_tracks.csv
```

### Step 3: Train the Emotion Model
Train the VGG19 Convolutional Neural Network on the FER-2013 dataset.
```bash
python src/emotion/train.py
```
*Note: This will take some time depending on your CPU/GPU. Once finished, it will save `emotion_model.keras` in the `models/` folder.*

### Step 4: Train the Recommendation Engine
Process the raw Spotify CSV and train the K-Nearest Neighbors (KNN) recommendation model.
```bash
python src/recommendation/recommender.py
```
*Note: This will clean the data, train the scaler and KNN model, and save `recommender.pkl` and `scaler.pkl` in the `models/` folder.*

### Step 5: Run the Application
Once both models are trained and saved, launch the Streamlit dashboard:
```bash
streamlit run app/app.py
```
Your browser will open automatically. Grant it permission to use your camera, capture your mood, and discover your personalized playlist!

---

## 📁 Directory Structure
```text
emotion-music-recommender/
├── app/
│   ├── app.py                     # Main Streamlit Application
│   └── components/
│       └── recommendation_ui.py   # UI components and feedback logic
├── data/
│   ├── processed/                 # Cleaned datasets
│   └── raw/                       # Raw Kaggle datasets
├── models/                        # Trained .keras and .pkl files
├── src/
│   ├── emotion/
│   │   ├── predict.py             # OpenCV detection & Model inference
│   │   └── train.py               # VGG19 training pipeline
│   ├── recommendation/
│   │   └── recommender.py         # KNN model training & prediction
│   └── utils/
│       └── config.py              # Centralized paths and mappings
├── requirements.txt
└── README.md
```

## ⚠️ Troubleshooting Virtual Cameras (OBS/Camo/DroidCam)
If you are using a virtual camera app on Windows and the browser displays a black or green screen:
1. Ensure the virtual camera app is running *before* opening the browser.
2. In your Chromium browser (Chrome/Brave/Edge), navigate to your camera settings (e.g., `brave://settings/content/camera`).
3. Select your virtual camera (e.g., OBS Virtual Camera) from the dropdown. 
4. If drivers fail, use the **"Or upload a photo"** fallback button provided in the dashboard to test the ML pipeline manually!