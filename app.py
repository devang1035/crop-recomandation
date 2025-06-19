import streamlit as st
import joblib
import numpy as np

# Load model and utilities
model = joblib.load("model/crop_model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

# Language dictionary
languages = {
    "English": {
        "title": "🌾 Crop Recommendation System",
        "submit": "Recommend Crop",
        "inputs": ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "Temperature (°C)", "Humidity (%)", "pH", "Rainfall (mm)"],
    },
    "हिंदी": {
        "title": "🌾 फसल सिफारिश प्रणाली",
        "submit": "फसल सुझाएं",
        "inputs": ["नाइट्रोजन (N)", "फॉस्फोरस (P)", "पोटैशियम (K)", "तापमान (°C)", "आर्द्रता (%)", "पीएच", "वर्षा (mm)"],
    },
    "ગુજરાતી": {
        "title": "🌾 પાક ભલામણ સિસ્ટમ",
        "submit": "પાક ભલામણ કરો",
        "inputs": ["નાઈટ્રોજન (N)", "ફોસ્ફરસ (P)", "પોટેશિયમ (K)", "તાપમાન (°C)", "ભેજ (%)", "pH", "વરસાદ (mm)"],
    }
}

# Language selector
selected_lang = st.sidebar.selectbox("🌐 Choose Language", list(languages.keys()))
text = languages[selected_lang]

st.title(text["title"])

# Vertical layout input fields
inputs = []
for label in text["inputs"]:
    value = st.number_input(label, step=0.1, format="%.2f")
    inputs.append(value)

# Predict button
if st.button(text["submit"]):
    input_scaled = scaler.transform([inputs])
    probs = model.predict_proba(input_scaled)[0]
    top3_idx = probs.argsort()[-3:][::-1]
    confidence = probs[top3_idx[0]]

    predicted_crop = label_encoder.inverse_transform([top3_idx[0]])[0]
    st.success(f"✅ Recommended Crop: {predicted_crop.capitalize()}")

    # Show top 3 crop predictions
    st.markdown("### 🔍 Top 3 Crop Suggestions:")
    for idx in top3_idx:
        crop_name = label_encoder.inverse_transform([idx])[0]
        st.write(f"➡️ {crop_name.capitalize()}: {probs[idx]*100:.2f}%")
