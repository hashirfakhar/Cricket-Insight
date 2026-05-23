"""
========================================================
Cricket Player Performance Predictor — Streamlit App
Section: 8 - Deployment
========================================================
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Cricket Performance Predictor",
    page_icon="🏏",
    layout="centered",
)

# ── Load model ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("models/best_model.pkl")
    scaler   = joblib.load("models/scaler.pkl")
    features = joblib.load("models/feature_names.pkl")
    return model, scaler, features

try:
    model, scaler, FEATURES = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model not found: {e}. Please run ml_pipeline.py first.")

# ── Career stats lookup (for auto-fill) ──────────────────────────
PLAYER_STATS = {
    "Virat Kohli":       {"avg": 52.73, "sr": 139.0, "matches": 115, "role": "Batsman"},
    "Rohit Sharma":      {"avg": 32.05, "sr": 140.0, "matches": 148, "role": "Batsman"},
    "KL Rahul":          {"avg": 45.59, "sr": 136.8, "matches": 72,  "role": "Batsman"},
    "Suryakumar Yadav":  {"avg": 46.42, "sr": 175.2, "matches": 64,  "role": "Batsman"},
    "Babar Azam":        {"avg": 41.13, "sr": 129.7, "matches": 108, "role": "Batsman"},
    "Mohammad Rizwan":   {"avg": 32.15, "sr": 132.4, "matches": 88,  "role": "WK-Batsman"},
    "Jos Buttler":       {"avg": 36.01, "sr": 143.0, "matches": 102, "role": "WK-Batsman"},
    "Glenn Maxwell":     {"avg": 33.40, "sr": 162.0, "matches": 80,  "role": "All-rounder"},
    "David Warner":      {"avg": 32.18, "sr": 144.4, "matches": 96,  "role": "Batsman"},
    "Kane Williamson":   {"avg": 33.60, "sr": 128.8, "matches": 107, "role": "Batsman"},
    "Rashid Khan":       {"avg": 17.00, "sr": 88.0,  "matches": 58,  "role": "Bowler"},
    "Shakib Al Hasan":   {"avg": 22.10, "sr": 125.0, "matches": 122, "role": "All-rounder"},
    "Custom Player":     {"avg": 30.0,  "sr": 130.0, "matches": 50,  "role": "Batsman"},
}

OPPOSITION_RANKS = {
    "India": 1, "England": 2, "Pakistan": 3, "Australia": 4,
    "New Zealand": 5, "South Africa": 6, "West Indies": 7,
    "Bangladesh": 8, "Afghanistan": 9, "Sri Lanka": 10,
}

VENUES = [
    "Wankhede Stadium", "MCG", "Lord's", "Eden Gardens",
    "Gaddafi Stadium", "SuperSport Park", "Kensington Oval",
    "Shere Bangla Stadium", "Sharjah Cricket Stadium",
    "Dubai International Stadium",
]

ROLES = {"Batsman": 0, "WK-Batsman": 1, "All-rounder": 2, "Bowler": 3}

# ── Header ───────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 1.5rem 0 0.5rem 0;'>
    <h1 style='font-size:2.4rem; margin-bottom:0;'>🏏 Cricket Performance Predictor</h1>
    <p style='color:#888; font-size:1rem; margin-top:0.3rem;'>
        Predict whether a T20I batsman will be a <b>Top Performer</b> (50+ runs)
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Sidebar — About ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Model:** XGBoost (Tuned)  
    **Task:** Binary classification  
    **Target:** Predicts if a player scores 50+ runs  
    **Accuracy:** ~99% on test set  
    **AUC-ROC:** 1.000  
    
    ---
    **Features used:**
    - Balls faced & Strike rate
    - Career average & SR
    - Dominance score (SR × BF)
    - Runs vs career average
    - Opposition difficulty
    - Player experience tier
    - Venue & opposition encoding
    
    ---
    *ML Final Project — 2024*
    """)

# ── Input Form ───────────────────────────────────────────────────
st.markdown("### 📋 Match Input")

col1, col2 = st.columns(2)

with col1:
    player_choice = st.selectbox(
        "Select Player",
        options=list(PLAYER_STATS.keys()),
        help="Select a known player to auto-fill career stats, or choose 'Custom Player'."
    )

    # Auto-fill career stats
    stats = PLAYER_STATS[player_choice]

    balls_faced = st.number_input(
        "Balls Faced", min_value=1, max_value=200, value=30,
        help="Number of balls the batsman faced in this innings."
    )
    strike_rate = st.number_input(
        "Strike Rate", min_value=0.0, max_value=300.0, value=130.0, step=0.5,
        help="Runs scored per 100 balls."
    )
    dismissed = st.selectbox(
        "Was the batsman dismissed?", options=["Yes", "No"],
        help="Whether the batsman got out."
    )

with col2:
    opposition = st.selectbox(
        "Opposition Team",
        options=list(OPPOSITION_RANKS.keys()),
    )
    venue = st.selectbox("Venue", options=VENUES)
    career_avg = st.number_input(
        "Career Batting Average", min_value=1.0, max_value=80.0,
        value=float(stats["avg"]), step=0.5,
    )
    career_sr = st.number_input(
        "Career Strike Rate", min_value=50.0, max_value=200.0,
        value=float(stats["sr"]), step=0.5,
    )
    t20i_matches = st.number_input(
        "T20I Matches Played", min_value=1, max_value=200,
        value=int(stats["matches"]),
    )
    player_role = st.selectbox(
        "Player Role",
        options=list(ROLES.keys()),
        index=list(ROLES.keys()).index(stats["role"]),
    )

# ── Input Validation ─────────────────────────────────────────────
errors = []
if balls_faced < 1:
    errors.append("❌ Balls faced must be at least 1.")
if strike_rate < 0 or strike_rate > 300:
    errors.append("❌ Strike rate must be between 0 and 300.")
if career_avg < 1:
    errors.append("❌ Career average must be at least 1.")

if errors:
    for e in errors:
        st.error(e)

# ── Feature Construction ─────────────────────────────────────────
def build_features(balls_faced, strike_rate, dismissed_str, opposition,
                   venue, career_avg, career_sr, t20i_matches, player_role):
    dismissed_val    = 1 if dismissed_str == "Yes" else 0
    opposition_rank  = OPPOSITION_RANKS.get(opposition, 5)
    opp_difficulty   = 1 / opposition_rank
    role_enc         = ROLES.get(player_role, 0)
    venue_enc        = VENUES.index(venue) if venue in VENUES else 0
    opposition_enc   = list(OPPOSITION_RANKS.keys()).index(opposition) if opposition in OPPOSITION_RANKS else 0

    # Estimated runs from SR and BF
    est_runs         = (strike_rate / 100) * balls_faced
    runs_vs_avg      = est_runs / (career_avg + 1)
    dominance_score  = (strike_rate * balls_faced) / 100
    balls_efficiency = est_runs / (balls_faced + 1)

    # Experience tier
    if t20i_matches <= 30:   exp_tier = 0
    elif t20i_matches <= 70: exp_tier = 1
    elif t20i_matches <= 120: exp_tier = 2
    else:                     exp_tier = 3

    row = {
        "dominance_score":  dominance_score,
        "runs_vs_avg":      runs_vs_avg,
        "balls_faced":      balls_faced,
        "balls_efficiency": balls_efficiency,
        "t20i_career_avg":  career_avg,
        "strike_rate":      strike_rate,
        "t20i_career_sr":   career_sr,
        "t20i_matches":     t20i_matches,
        "venue_enc":        venue_enc,
        "opposition_rank":  opposition_rank,
    }
    return pd.DataFrame([row])

# ── Predict Button ────────────────────────────────────────────────
st.divider()

if st.button("🔍 Predict Performance", type="primary", use_container_width=True, disabled=not model_loaded or bool(errors)):
    X_input = build_features(
        balls_faced, strike_rate, dismissed, opposition,
        venue, career_avg, career_sr, t20i_matches, player_role
    )

    # Align columns
    X_input = X_input[FEATURES]

    prediction   = model.predict(X_input)[0]
    probability  = model.predict_proba(X_input)[0]
    confidence   = probability[int(prediction)] * 100

    st.divider()
    st.markdown("### 🎯 Prediction Result")

    if prediction == 1:
        st.success(f"### ✅ TOP PERFORMER — {confidence:.1f}% confidence")
        st.markdown(f"""
        The model predicts **{player_choice}** is likely to score **50+ runs** in this innings.

        | Feature | Value |
        |---|---|
        | Estimated Runs | {(strike_rate/100)*balls_faced:.0f} |
        | Dominance Score | {(strike_rate*balls_faced/100):.1f} |
        | Opposition | {opposition} (Rank #{OPPOSITION_RANKS[opposition]}) |
        | Venue | {venue} |
        | Experience | {t20i_matches} T20Is |
        """)
    else:
        st.warning(f"### ❌ NOT a Top Performer — {confidence:.1f}% confidence")
        st.markdown(f"""
        The model predicts **{player_choice}** is **unlikely to score 50+** in this innings.

        | Feature | Value |
        |---|---|
        | Estimated Runs | {(strike_rate/100)*balls_faced:.0f} |
        | Dominance Score | {(strike_rate*balls_faced/100):.1f} |
        | Opposition | {opposition} (Rank #{OPPOSITION_RANKS[opposition]}) |
        | Venue | {venue} |
        | Experience | {t20i_matches} T20Is |
        """)

    # Confidence bar
    st.markdown("**Confidence Breakdown:**")
    col_a, col_b = st.columns(2)
    col_a.metric("Not Top Performer", f"{probability[0]*100:.1f}%")
    col_b.metric("Top Performer",     f"{probability[1]*100:.1f}%")

    st.progress(int(probability[1] * 100))

    st.info("""
    **How the model decides:**  
    The XGBoost model weighs dominance score (SR × balls faced), career average, 
    balls efficiency, and opposition difficulty to estimate the probability of a 
    50+ innings. These features together capture both player quality and match context.
    """)

# ── Footer ───────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.8rem;'>"
    "ML Final Project | Cricket Performance Prediction | XGBoost Model"
    "</div>",
    unsafe_allow_html=True,
)