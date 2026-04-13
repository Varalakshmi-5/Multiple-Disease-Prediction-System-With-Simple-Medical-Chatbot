import streamlit as st
import pandas as pd
import joblib


# ======================================================
# BACKGROUND STYLE ONLY (SAFE)
# ======================================================
def set_diabetes_bg_color():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #C6F4D6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# ======================================================
# MAIN PAGE
# ======================================================
def diabetes_prediction_page():

    set_diabetes_bg_color()

    model = joblib.load("model/diabetes_model.pkl")

    # --------------------------------------------------
    # Title
    # --------------------------------------------------
    st.markdown(
        """
        <h2 style='text-align:center; color:#1b5e20;'>🩸 Diabetes Prediction System</h2>
        <p style='text-align:center; color:#2e7d32;'>
            Enter patient details to predict diabetes risk
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # --------------------------------------------------
    # Inputs
    # --------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        gender          = st.selectbox("Gender", ["Male", "Female"])
        age             = st.number_input("Age", 1, 120, 30)
        hypertension    = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease   = st.selectbox("Heart Disease", ["No", "Yes"])

    with col2:
        smoking_history = st.selectbox(
            "Smoking History",
            ["never", "former", "current", "not current", "ever"]
        )
        bmi     = st.number_input("BMI", 10.0, 60.0, 25.0)
        hba1c   = st.number_input("HbA1c Level", 3.0, 15.0, 5.5)
        glucose = st.number_input("Blood Glucose Level", 50, 300, 120)

    st.divider()

    # --------------------------------------------------
    # Encoding
    # --------------------------------------------------
    gender_map  = {"Male": 1, "Female": 0}
    smoking_map = {
        "never": 0, "former": 1, "current": 2,
        "not current": 3, "ever": 4
    }

    input_data = pd.DataFrame(
        [[
            gender_map[gender],
            age,
            1 if hypertension == "Yes" else 0,
            1 if heart_disease == "Yes" else 0,
            smoking_map[smoking_history],
            bmi,
            hba1c,
            glucose
        ]],
        columns=[
            "gender", "age", "hypertension", "heart_disease",
            "smoking_history", "bmi", "hbA1c_level", "blood_glucose_level"
        ]
    )

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    if st.button("🔍 Predict Diabetes"):

        prediction = model.predict(input_data)[0]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_data)[0][1]

            if prediction == 1:
                disease    = "Diabetic"
                confidence = f"{prob:.1%}"
                st.error(f"⚠️ High Risk of Diabetes\n\nRisk Probability: {prob:.2f}")
            else:
                disease    = "Non-Diabetic"
                confidence = f"{1 - prob:.1%}"
                st.success(f"✅ Low Risk of Diabetes\n\nSafety Probability: {1 - prob:.2f}")
        else:
            disease    = "Diabetic" if prediction == 1 else "Non-Diabetic"
            confidence = "N/A"
            st.success("✅ Prediction completed")

        # ── Save result so app.py can log it to the health records DB ──
        st.session_state["last_diabetes_result"] = {
            "disease":    disease,
            "confidence": confidence,
            "symptoms":   [],                          # diabetes has no symptom input
            "details": {
                "age":              age,
                "gender":           gender,
                "bmi":              bmi,
                "hba1c":            hba1c,
                "blood_glucose":    glucose,
                "hypertension":     hypertension,
                "heart_disease":    heart_disease,
                "smoking_history":  smoking_history,
            }
        }
        # Reset the "already saved" flag so app.py saves this fresh result
        st.session_state["diabetes_saved"] = False