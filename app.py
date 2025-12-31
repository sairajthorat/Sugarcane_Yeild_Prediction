import streamlit as st
import pandas as pd
import joblib
import os

# Page Config
st.set_page_config(
    page_title="Sugarcane Yield Predictor (Farmer)",
    page_icon="🌾",
    layout="wide"
)

# Constants & Paths
# Using relative paths for portability within Test004
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sugarcane_yield_model.pkl')
COLUMNS_PATH = os.path.join(BASE_DIR, 'models', 'model_columns.pkl')

# --- Taluka Weather Mapping (Securely Loaded) ---
try:
    TALUKA_WEATHER_MAP = st.secrets["taluka_weather"]
except FileNotFoundError:
    st.error("Secrets file not found! Please ensure .streamlit/secrets.toml exists.")
    st.stop()
except KeyError:
    st.error("Key 'taluka_weather' not found in secrets!")
    st.stop()


# Load Artifacts
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        return None, None
    model = joblib.load(MODEL_PATH)
    cols = joblib.load(COLUMNS_PATH)
    return model, cols

model, model_columns = load_model()

# Title
st.title("🌾 Farmer Yield Predictor")
st.markdown("Select your farm details. Weather conditions are automatically fetched based on your location.")

if not model:
    st.stop()

# --- Layout ---
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.header("1. Farm Details")
    
    # Input Fields
    taluka = st.selectbox("Taluka (Location)", list(TALUKA_WEATHER_MAP.keys()))
    season = st.selectbox("Season", ['Suru', 'Pre-seasonal', 'Adsali'])
    variety = st.selectbox("Cane Variety", ['CoM 0265', 'Co 86032', 'Co 92005', 'VSI 434'])
    soil = st.selectbox("Soil Type", ['Black Cotton', 'Clay Loam', 'Sandy Loam', 'Medium Black'])
    irrigation = st.selectbox("Irrigation Method", ['Drip', 'Flood', 'Rainfed'])

    # Retrieve Weather Data directly
    weather_data = TALUKA_WEATHER_MAP[taluka]

with col2:
    st.header("2. Environmental Conditions")
    st.info(f"ℹ️ Typical conditions for **{taluka}** during this period:")
    
    # Display Stats as Metrics
    c1, c2 = st.columns(2)
    c1.metric("🌧️ Rainfall", f"{weather_data['rainfall']} mm")
    c2.metric("🌡️ Max Temp", f"{weather_data['max_temp']} °C")
    
    c3, c4 = st.columns(2)
    c3.metric("💧 Humidity", f"{weather_data['humidity']}%")
    c4.metric("☀️ Solar Rad", f"{weather_data['solar']} kWh/m²")

    # Hidden details view
    #with st.expander("See full technical details"):
    #   st.write(weather_data)
    
# --- Prediction Logic ---
if st.button("🌱 Predict Yield", type="primary", use_container_width=True):
    
    # Prepare Input
    input_data = {
        'Taluka': taluka,
        'Season': season,
        'Cane_Variety': variety,
        'Soil_Type': soil,
        'Irrigation_Method': irrigation,
        'Latitude': weather_data['lat'],
        'Longitude': weather_data['lon'],
        'Area_Harvested_Ha': 1.0, 
        'Avg_NDVI': weather_data['ndvi'],
        'Avg_EVI': weather_data['ndvi'] * 0.8,
        'Avg_LST_Celsius': weather_data['max_temp'] - 5,
        'Avg_Max_Temp_Celsius': weather_data['max_temp'],
        'Avg_Min_Temp_Celsius': weather_data['min_temp'],
        'Avg_Humidity_Percent': weather_data['humidity'],
        'Solar_Radiation_kWh': weather_data['solar'],
        'Accumulated_Rainfall_mm': weather_data['rainfall']
    }
    
    # Process
    input_df = pd.DataFrame([input_data])
    input_df_encoded = pd.get_dummies(input_df, columns=['Taluka', 'Season', 'Cane_Variety', 'Soil_Type', 'Irrigation_Method'])
    input_df_final = input_df_encoded.reindex(columns=model_columns, fill_value=0)
    
    # Predict
    try:
        pred = model.predict(input_df_final)[0]
        st.success(f"### Estimated Yield: {pred:.2f} Tonnes/Ha")
        

    except Exception as e:
        st.error(f"Prediction failed: {e}")
