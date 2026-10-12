import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PROJECT IMPORTS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.emotion.predict import EmotionPredictor
from src.recommendation.recommender import MusicRecommender
from components.recommendation_ui import (
    render_recommendation_card,
    save_user_feedback,
)

# --------------------------------------------------
# STREAMLIT CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Emotion-Based Music Recommender",
    page_icon="🎶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():
    """Load the emotion predictor and music recommender once."""
    predictor = EmotionPredictor()

    recommender = MusicRecommender()
    recommender.load_model()

    return predictor, recommender


try:
    predictor, recommender = load_models()
except Exception as exc:
    st.error(f"Could not load the project models: {exc}")
    st.stop()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "current_emotion" not in st.session_state:
    st.session_state["current_emotion"] = "happy"

if "current_probabilities" not in st.session_state:
    st.session_state["current_probabilities"] = {
        "happy": 0.8,
        "neutral": 0.2,
    }

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ Control Panel")

user_id = st.sidebar.text_input(
    "User ID",
    value="U001",
)

genre_filter = st.sidebar.selectbox(
    "Filter by Genre",
    [
        "All",
        "Pop",
        "Rock",
        "Hip-Hop",
        "Jazz",
        "Classical",
        "Electronic",
    ],
)

top_n_songs = st.sidebar.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10,
)

# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.title("🎶 Emotion-Based Music Recommendation System")

st.markdown(
    "Detect your facial emotion and receive music "
    "recommendations tailored to your mood using machine learning."
)

col_cam, col_info = st.columns([1.1, 0.9])

# --------------------------------------------------
# CAMERA INPUT AND PHOTO UPLOAD
# --------------------------------------------------

with col_cam:
    st.subheader("📷 Camera / Photo Input")

    st.info(
        "Capture a photo using your selected browser camera, "
        "or upload a JPG or PNG image."
    )

    cam_buffer = st.camera_input(
        "Capture your expression",
        key="emotion_camera",
    )

    upload_buffer = st.file_uploader(
        "Or upload a photo",
        type=["jpg", "jpeg", "png"],
        key="emotion_upload",
    )

# --------------------------------------------------
# EMOTION ANALYSIS DISPLAY
# --------------------------------------------------

with col_info:
    st.subheader("🧠 Emotion Analysis")

    emotion_placeholder = st.empty()
    confidence_placeholder = st.empty()
    prob_chart_placeholder = st.empty()

# --------------------------------------------------
# IMAGE DECODING AND EMOTION PREDICTION
# --------------------------------------------------

# Prefer a newly captured photo over an uploaded photo.
img_buffer = (
    cam_buffer
    if cam_buffer is not None
    else upload_buffer
)

if img_buffer is not None:
    try:
        # Read the uploaded or captured image bytes.
        bytes_data = img_buffer.getvalue()

        if not bytes_data:
            st.error(
                "The selected image is empty. "
                "Please capture another photo or upload an image."
            )
        else:
            # Decode image bytes into an OpenCV BGR image.
            image_array = np.frombuffer(
                bytes_data,
                dtype=np.uint8,
            )

            cv2_img = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR,
            )

            # Check whether OpenCV decoded the image.
            if cv2_img is None or cv2_img.size == 0:
                st.error(
                    "OpenCV could not decode the image. "
                    "Please try another photo or upload a valid "
                    "JPG or PNG image."
                )

            else:
                # Display the exact image received by Python.
                st.markdown("**Image received by the application**")

                st.image(
                    cv2.cvtColor(
                        cv2_img,
                        cv2.COLOR_BGR2RGB,
                    ),
                    channels="RGB",
                    caption="Image received from the camera or uploader",
                    use_container_width=True,
                )

                # Check whether the received image is almost
                # a single solid colour, such as a green frame.
                channel_std = np.std(cv2_img)

                if channel_std < 5:
                    st.warning(
                        "The received image appears almost entirely "
                        "one colour. This may indicate a camera-feed "
                        "or virtual-camera problem. Try uploading a "
                        "normal photo to test the emotion model."
                    )

                else:
                    # Run the existing emotion prediction model.
                    results = predictor.detect_and_predict(cv2_img)

                    if results:
                        result = results[0]

                        emotion = result["emotion"]
                        probabilities = result["probabilities"]

                        st.session_state["current_emotion"] = emotion
                        st.session_state[
                            "current_probabilities"
                        ] = probabilities

                        st.success(
                            f"Emotion detected: {emotion.upper()}"
                        )

                    else:
                        st.warning(
                            "No face was detected. Please face the "
                            "camera, improve the lighting, or upload "
                            "a clearer photo."
                        )

    except Exception as exc:
        st.error(
            f"Image processing or emotion prediction failed: {exc}"
        )

# --------------------------------------------------
# UPDATE EMOTION METRICS
# --------------------------------------------------

detected_emotion = st.session_state["current_emotion"]

probabilities = st.session_state["current_probabilities"]

confidence_score = (
    float(probabilities.get(detected_emotion, 0.0)) * 100
)

with col_info:
    emotion_placeholder.markdown(
        f"### Detected Mood: **{detected_emotion.upper()}**"
    )

    confidence_placeholder.progress(
        min(100, max(0, int(confidence_score))),
        text=f"Confidence: {confidence_score:.1f}%",
    )

    probability_df = pd.DataFrame(
        list(probabilities.items()),
        columns=["Emotion", "Probability"],
    )

    prob_chart_placeholder.bar_chart(
        probability_df.set_index("Emotion")
    )

# --------------------------------------------------
# MUSIC RECOMMENDATIONS
# --------------------------------------------------

st.markdown("---")
st.subheader("🎧 Recommended For You")

try:
    recommendations = recommender.recommend_for_emotion(
        detected_emotion,
        top_n=top_n_songs,
        genre_filter=genre_filter,
    )

    if recommendations is None or len(recommendations) == 0:
        st.info(
            "No songs were found for this mood and genre. "
            "Try selecting All in the genre filter."
        )

    else:
        for index, song in enumerate(recommendations, start=1):
            render_recommendation_card(song, index)

            cols = st.columns([1, 1, 1, 6])

            if cols[0].button(
                "❤️ Like",
                key=f"like_{index}",
            ):
                save_user_feedback(
                    user_id,
                    song["track_name"],
                    song["artist_name"],
                    detected_emotion,
                    "Liked",
                )

                st.toast(
                    f"Saved 'Liked' feedback for "
                    f"{song['track_name']}!"
                )

            if cols[1].button(
                "👎 Dislike",
                key=f"dislike_{index}",
            ):
                save_user_feedback(
                    user_id,
                    song["track_name"],
                    song["artist_name"],
                    detected_emotion,
                    "Disliked",
                )

                st.toast("Saved 'Disliked' feedback!")

            if cols[2].button(
                "⏭ Skip",
                key=f"skip_{index}",
            ):
                save_user_feedback(
                    user_id,
                    song["track_name"],
                    song["artist_name"],
                    detected_emotion,
                    "Skipped",
                )

                st.toast(
                    f"Skipped {song['track_name']}!"
                )

except Exception as exc:
    st.error(
        f"Could not generate music recommendations: {exc}"
    )
