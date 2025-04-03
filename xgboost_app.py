import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

st.set_page_config(page_title="XGBoost Player Value Predictor", layout="wide")

st.markdown("""
<h1 style='text-align: center; color: #4B84FF;'>⚽️ Football Player Value Estimator</h1>
<hr>
<p style='text-align: center;'>Estimate player market value using advanced AI trained on football statistics and attributes.</p>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("xgboost_model.pkl")

model = load_model()

# Load player predictions from precomputed CSV
@st.cache_data
def load_precomputed_predictions():
    try:
        df = pd.read_csv("outputs/PlayerValuePredictions.csv")
        return df
    except:
        return pd.DataFrame()

precomputed_df = load_precomputed_predictions()

# DEBUG: Display the first few rows of precomputed data
# st.write(precomputed_df.head())

# -------------------- Search Player -------------------- #
st.markdown("""
<h2 style='color:#4a90e2;'>🔎 Search Player Value</h2>
<p>Look up predicted values for players already stored in the dataset.</p>
""", unsafe_allow_html=True)

if not precomputed_df.empty and 'player_name' in precomputed_df.columns:
    player_names = precomputed_df['player_name'].dropna().unique().tolist()
    search_name = st.text_input("Enter player name")
else:
    st.warning("⚠️ No player data available for search.")
    search_name = ""


if search_name and not precomputed_df.empty:
    result = precomputed_df[precomputed_df['player_name'].str.lower().str.contains(search_name.lower())]
    if not result.empty:
        st.success(f"Found {len(result)} matching player(s):")
        st.dataframe(result[['player_name', 'actual_value', 'predicted_value']])
    else:
        st.warning("No player found with that name.")

# UI Tabs
tab1, tab2, tab3 = st.tabs(["Manual Input", "Batch Prediction", "EDA Report"])

# -------------------- Tab 1: Manual Input -------------------- #
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔧 Enter Player Attributes")
        st.markdown("#### ✨ Skill Attributes")
        overall = st.slider("Overall", 40, 99, 75, help="Player's current overall rating")
        potential = st.slider("Potential", 40, 99, 80, help="Player's growth ceiling")
        skill_moves = st.slider("Skill Moves", 1, 5, 3, help="1 to 5 star skill level")
        weak_foot = st.slider("Weak Foot", 1, 5, 3, help="How good is the weaker foot")

        st.markdown("#### 💪 Physical Attributes")
        age = st.slider("Age", 16, 45, 25, help="Player's age")
        height_cm = st.slider("Height (cm)", 150, 210, 180, help="Height in centimeters")
        weight_kg = st.slider("Weight (kg)", 50, 120, 75, help="Weight in kilograms")

        st.markdown("#### 🌍 Reputation & Wage")
        wage = st.number_input("Wage (€)", 1000, 1000000, 12000, step=1000, help="Weekly wage in Euros (default ≈ average wage)")
        intl_rep = st.slider("International Reputation", 1, 5, 2, help="Reputation from 1 (low) to 5 (top)")

        position_options = ['CB', 'CDM', 'CF', 'CM', 'GK', 'LAM', 'LB', 'LCB', 'LCM', 'LDM', 'LF', 'LM', 'LS',
                            'LW', 'LWB', 'RAM', 'RB', 'RCB', 'RCM', 'RDM', 'RF', 'RM', 'RS', 'RW', 'RWB', 'ST']
        position = st.selectbox("Position", position_options, help="Primary playing position")

    with col2:
        st.subheader("💰 Prediction Result")
        if st.button("🎯 Predict Value"):
            features = {
                "overall": overall,
                "potential": potential,
                "wage": wage,
                "international_reputation": intl_rep,
                "skill_moves": skill_moves,
                "weak_foot": weak_foot,
                "age": age,
                "height_cm": height_cm,
                "weight_kg": weight_kg
            }

            for pos in position_options:
                features[f"position_{pos}"] = 1 if pos == position else 0

            input_df = pd.DataFrame([features])
            predicted_log = model.predict(input_df)[0]
            predicted_value = np.expm1(predicted_log)

            st.markdown(f"""
            <div style='padding: 20px; background-color: #eafaf1; border-left: 5px solid #28a745; border-radius: 8px;'>
                <h3 style='color: #155724;'>Estimated Market Value:</h3>
                <h1 style='color: #28a745;'>€{predicted_value:,.0f}</h1>
            </div>
            """, unsafe_allow_html=True)
            

# -------------------- Tab 2: Batch Upload -------------------- #
with tab2:
    st.subheader("📤 Upload Player Data")
    uploaded_file = st.file_uploader("Upload CSV with full model columns", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if all(col in df.columns for col in model.feature_names_in_):
            df['predicted_log'] = model.predict(df[model.feature_names_in_])
            df['predicted_value'] = np.expm1(df['predicted_log'])

            st.markdown("### ✅ Predictions")
            st.dataframe(df[['name'] + list(model.feature_names_in_) + ['predicted_value']] if 'name' in df.columns else df)

            # Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions", csv, file_name="predicted_players.csv", mime="text/csv")
        else:
            st.error("CSV missing required model input columns. Please include all trained features.")

# -------------------- Tab 3: EDA -------------------- #
with tab3:
    st.subheader("📊 Exploratory Data Analysis")
    try:
        with open("notebooks/eda_report.html", 'r', encoding='utf-8') as f:
            eda_html = f.read()
        st.components.v1.html(eda_html, height=1000, scrolling=True)
    except FileNotFoundError:
        st.warning("EDA report not found. Please generate eda_report.html using Jupyter nbconvert.")
