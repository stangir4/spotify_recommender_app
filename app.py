# app.py
# Spotify behavior-based recommender (lightweight, rule-based + cluster profile viewer)
# Designed to run on Streamlit using the repository artifacts folder.

import streamlit as st
import json
import os

st.set_page_config(page_title="Spotify Behavior Recommender", page_icon="🎧", layout="wide")
st.title("🎧 Spotify Behavior-based Playlist Recommender")
st.markdown("Give a few quick inputs and I’ll recommend a playlist based on time-of-day, mood, and genre. "
            "If cluster profiles are available, you'll also see a short explanation of the user cluster(s).")

# --- Load cluster profiles if available ---
cluster_profiles = {}
profiles_path = "artifacts/cluster_profiles.json"
if os.path.exists(profiles_path):
    try:
        with open(profiles_path, "r") as f:
            cluster_profiles = json.load(f)
    except Exception as e:
        st.warning(f"Could not load cluster_profiles.json: {e}")

# --- Simple rule-base for recommendations (fallback + explainability) ---
# These are example playlists; replace with real playlist URIs if you want.
playlist_rules = [
    # (time_slots, moods, genres) -> playlist name
    (["Morning", "Early Morning"], ["Energetic", "Upbeat"], ["Pop", "EDM", "Dance"], "🏃 Morning Workout Mix"),
    (["Morning"], ["Calm", "Focused"], ["Classical", "Ambient"], "☕ Focused Morning"),
    (["Afternoon"], ["Relaxed", "Chill"], ["Lo-fi", "Indie"], "🌤️ Afternoon Chill"),
    (["Evening", "Night"], ["Sadness or melancholy", "Melancholic", "Reflective"], ["Acoustic", "Folk"], "🌙 Evening Reflective"),
    (["Night"], ["Excited", "Upbeat"], ["Hip Hop", "Rap", "Pop"], "🌃 Night Out Vibes"),
    (["Any"], ["Any"], ["Podcast"], "🎙️ Popular Podcasts"),
]

def recommend_playlist(time_slot, mood, genre):
    # exact matches first
    for t_slots, moods, genres, playlist in playlist_rules:
        if (time_slot in t_slots or "Any" in t_slots) and (mood in moods or "Any" in moods):
            # check genre overlap or accept Any
            if any(g.lower() in genre.lower() for g in genres) or "Podcast" in genres and "pod" in genre.lower():
                return playlist
    # fallback by genre keywords
    g = genre.lower()
    if "workout" in g or "upbeat" in g or "energetic" in g:
        return "🏃 High Energy Workout"
    if "lo-fi" in g or "chill" in g or "indie" in g:
        return "🌿 Lo-fi & Chill"
    if "podcast" in g or "talk" in g:
        return "🎧 Podcast Highlights"
    # default fallback
    return "🎶 Popular Hits"

# --- Sidebar inputs ---
with st.sidebar:
    st.header("Your Listening Context")
    age = st.selectbox("Age group", ["12-20", "20-35", "35-60", "60+"], index=1)
    gender = st.selectbox("Gender", ["Female", "Male", "Other", "Prefer not to say"])
    time_slot = st.selectbox("Time of day", ["Morning", "Early Morning", "Afternoon", "Evening", "Night"])
    mood = st.selectbox("Mood / Influential situation", [
        "Upbeat", "Energetic", "Relaxed", "Calm", "Sadness or melancholy", "Reflective", "Excited", "Any"
    ], index=7)
    genre = st.text_input("Favorite genre(s) (comma-separated)", "Pop, Melody")
    podcast_freq = st.selectbox("Podcast listen frequency", ["Never", "Monthly", "Weekly", "Daily"])

# Main UI: recommendation and cluster profile
st.subheader("Recommendation")
rec = recommend_playlist(time_slot, mood, genre)
st.markdown(f"### Recommended Playlist: **{rec}**")
st.write("**Why this recommendation?** — simple rule-based matching using time-of-day, mood, and genre keywords.")
st.info(f"Inputs: time={time_slot} · mood={mood} · genre={genre} · age={age}")

# Show cluster profiles if available
if cluster_profiles:
    st.subheader("Cluster Profiles (from KMeans centers)")
    st.write("These are the top features (from cluster centers) that define each cluster.")
    # cluster_profiles keys might be strings — ensure sorted numeric order if possible
    try:
        keys = sorted(cluster_profiles.keys(), key=lambda x: int(x))
    except Exception:
        keys = list(cluster_profiles.keys())
    cols = st.columns(len(keys))
    for i, k in enumerate(keys):
        with cols[i]:
            st.markdown(f"**Cluster {k}**")
            feats = cluster_profiles.get(k) or cluster_profiles.get(str(k)) or []
            if isinstance(feats, dict):
                # if the profile saved as feature->score, show top keys
                feats_list = list(feats.keys())[:10]
            else:
                feats_list = feats[:10]
            if feats_list:
                for f in feats_list:
                    st.write("•", f)
            else:
                st.write("_No top features found_")
else:
    st.info("No cluster_profiles.json found in artifacts/. If you'd like, generate cluster profiles in your notebook and re-upload the file.")

# small footer
st.markdown("---")
st.markdown("**Notes:** This app uses a lightweight rule-based recommender for the demo. "
            "You can replace the logic with a model-based predictor using the saved `kmeans_model.joblib` or other artifacts.")
