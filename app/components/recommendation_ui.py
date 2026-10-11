import streamlit as st
import pandas as pd
from datetime import datetime
import os

def render_recommendation_card(song, index):
    """Renders a single recommended song with audio characteristics and feedback actions."""
    with st.container():
        cols = st.columns([0.6, 0.3, 0.1])
        with cols[0]:
            st.markdown(f"**{index}. {song['track_name']}**")
            st.caption(f"Artist: {song['artist_name']} | Genre: {song.get('genre', 'Various')}")
        with cols[1]:
            # Mini feature metrics badge
            st.text(f"Energy: {song['energy']:.2f} | Valence: {song['valence']:.2f}")
        with cols[2]:
            st.markdown("🎵")
    st.divider()

def save_user_feedback(user_id, song_name, artist_name, emotion, action):
    """Saves user interaction feedback for personalization / history tracking."""
    feedback_file = "data/processed/user_feedback.csv"
    os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame([{
        'user_id': user_id,
        'track_name': song_name,
        'artist_name': artist_name,
        'emotion': emotion,
        'action': action,
        'timestamp': timestamp
    }])
    
    if os.path.exists(feedback_file):
        df = pd.read_csv(feedback_file)
        df = pd.concat([df, new_entry], ignore_index=True)
    else:
        df = new_entry
        
    df.to_csv(feedback_file, index=False)