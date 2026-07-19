import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# Custom styling - Black & Red theme
st.markdown("""
    <style>
    .stApp {
        background-color: #0d0d0d;
        color: #f5f5f5;
    }
    .main {
        background-color: #0d0d0d;
    }
    h1, h3, p, label {
        color: #f5f5f5 !important;
    }
    .stButton>button {
        background-color: #ff1c1c;
        color: white;
        border: none;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #b30000;
        transform: scale(1.02);
    }
    div[data-baseweb="select"], .stNumberInput input, .stSlider {
        background-color: #1a1a1a;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("SVM_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.markdown("<h1 style='text-align:center; color:#ff1c1c;'>❤️ Heart Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#cccccc;'>Enter patient details below</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 100, 50)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar >120", [0, 1])

with col2:
    restecg = st.selectbox("Rest ECG", [0, 1, 2])
    thalach = st.slider("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Angina", [0, 1])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

if st.button("🔍 Predict"):
    input_data = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }])

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    st.markdown("---")
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")