import streamlit as st
import numpy as np
import joblib
import base64
import os

# ======================================================
# FEATURE ORDER (MATCH TRAINING)
# ======================================================
FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

# ======================================================
# BACKGROUND IMAGE WITH DARK OVERLAY
# ======================================================
def set_background(image_path):
    if not os.path.exists(image_path):
        return

    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0, 0, 0, 0.55);
            z-index: -1;
        }}
        h1, h2, h3, h4, h5, h6, label, p {{
            color: black !important;
        }}
        div[data-baseweb="input"],
        div[data-baseweb="select"] {{
            background-color: rgba(255,255,255,0.9);
            border-radius: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def load_model():
    return joblib.load("model/heart_disease_model.joblib")

# ======================================================
# MAIN PAGE
# ======================================================
def heart_prediction_page():

    set_background("background1.jpg")

    st.markdown(
        "<h1 style='text-align:center;'>❤️ Heart Disease Prediction</h1>",
        unsafe_allow_html=True
    )

    model = load_model()

    with st.form("heart_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age      = st.number_input("Age", 1, 120, 45)
            sex      = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
            cp       = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
            trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)

        with col2:
            chol    = st.number_input("Cholesterol", 100, 600, 200)
            fbs     = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
            restecg = st.selectbox("Rest ECG", [0, 1, 2])
            thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)

        with col3:
            exang   = st.selectbox("Exercise Induced Angina", [0, 1])
            oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
            slope   = st.selectbox("Slope", [0, 1, 2])
            ca      = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4])
            thal    = st.selectbox("Thalassemia", [0, 1, 2, 3])

        submit = st.form_submit_button("🔍 Predict")

    if submit:
        input_data = np.array([[
            age, sex, cp, trestbps, chol, fbs,
            restecg, thalach, exang, oldpeak,
            slope, ca, thal
        ]])

        prediction = model.predict(input_data)[0]
        prob       = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            disease    = "Heart Disease Detected"
            confidence = f"{prob:.1%}"
            st.error(f"⚠️ High Risk of Heart Disease\n\nProbability: {prob:.2%}")
        else:
            disease    = "No Heart Disease"
            confidence = f"{1 - prob:.1%}"
            st.success(f"✅ No Heart Disease Detected\n\nProbability: {prob:.2%}")

        # ── Save result so app.py can log it to the health records DB ──
        st.session_state["last_heart_result"] = {
            "disease":    disease,
            "confidence": confidence,
            "symptoms":   [],
            "details": {
                "age":             age,
                "sex":             "Male" if sex == 1 else "Female",
                "chest_pain_type": cp,
                "resting_bp":      trestbps,
                "cholesterol":     chol,
                "fasting_bs":      "Yes" if fbs == 1 else "No",
                "rest_ecg":        restecg,
                "max_heart_rate":  thalach,
                "exang":           "Yes" if exang == 1 else "No",
                "st_depression":   oldpeak,
                "slope":           slope,
                "major_vessels":   ca,
                "thalassemia":     thal,
            }
        }
        # Reset saved flag so app.py logs this fresh result
        st.session_state["heart_saved"] = False