import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBRegressor

pipe = pickle.load(open('pipe.pkl', 'rb'))

teams = ['Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa',
         'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka']

cities = ['Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town',
          'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban',
          'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion',
          'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton',
          'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi', 'Nagpur',
          'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 'Cardiff', 'Christchurch', 'Trinidad']

st.markdown("""
    <style>
        body {
            background-color: white;  /* Set background color to white */
            font-family: 'Arial', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        body {
            background-color: white;  # Set background to white
            font-family: 'Arial', sans-serif;
        }

        .main .block-container {
            padding-top: 2rem;
        }

        h1 {
            color: #1D3557;
            font-size: 4rem;
            text-align: center;
            font-family: 'Arial', sans-serif;
            margin-bottom: 2rem;
        }

        .stButton>button {
            background-color: #FF6600;
            color: white;
            border-radius: 10px;
            padding: 10px 24px;
            font-size: 18px;
        }

        .stSelectbox>label, .stNumberInput>label {
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }

        .stSelectbox, .stNumberInput {
            margin-top: 1.5rem;
        }

        .stButton>button:hover {
            background-color: #FF4500;
        }

        .header-text {
            color: #4A90E2;
            font-size: 2rem;
            text-align: center;
            margin-bottom: 1rem;
        }

        .footer-text {
            font-size: 14px;
            text-align: center;
            color: #A0A0A0;
            margin-top: 4rem;
        }

        .stFileUploader {
            margin-top: 1rem;
        }
        
    </style>
""", unsafe_allow_html=True)

st.title('**Sports Analytics: T20 Score Prediction Engine**')

st.markdown("<h1 style='color: #FF6600;'>Rahul Purohit</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 18px; text-align: center; color: #333;'>Minor Project - Cricket Score Prediction using Machine Learning</p>", unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select Batting Team', sorted(teams), help="Choose the team currently batting")
with col2:
    bowling_team = st.selectbox('Select Bowling Team', sorted(teams), help="Choose the team currently bowling")

city = st.selectbox('Select City', sorted(cities), help="Select the city where the match is happening")

col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input('Current Score', min_value=0, help="Enter the current score of the batting team")
with col4:
    overs = st.number_input('Overs Done (works for overs >5)', min_value=0.0, max_value=20.0, step=0.1, help="Enter the number of overs completed")
with col5:
    wickets = st.number_input('Wickets Out', min_value=0, max_value=10, help="Enter the number of wickets fallen")

last_five = st.number_input('Runs Scored in Last 5 Overs', min_value=0, help="Enter the number of runs scored in the last 5 overs")

st.markdown("---")

if st.button('Predict Score'):
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = current_score / overs

    input_df = pd.DataFrame(
        {'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': city,
         'current_score': [current_score], 'balls_left': [balls_left], 'wickets_left': [wickets],
         'crr': [crr], 'last_five': [last_five]})

    result = pipe.predict(input_df)

    st.header(f"Predicted Score: {int(result[0])} 🏏")

    st.write("This prediction is based on a machine learning model trained on historical match data. The model evaluates current match conditions and predicts the score in the remaining overs.")

st.markdown("---")
st.markdown("<div class='footer-text'>Project by Rahul Purohit | Minor Project Submission</div>", unsafe_allow_html=True)
