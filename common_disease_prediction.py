import streamlit as st
import joblib
import pandas as pd
import numpy as np
import base64


def common_disease_page():

    # ---------- BACKGROUND ----------
    def set_bg(image_file):
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    set_bg("background.jpg")

    # ---------- LOAD MODEL ----------
    model = joblib.load(r"model/disease_prediction_model.pkl")
    le    = joblib.load(r"model/label_encoder.pkl")

    df = pd.read_csv(r"datasets/Training.csv")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    all_symptoms = df.drop("prognosis", axis=1).columns.tolist()

    # ---------- UI ----------
    st.markdown(
        "<h1 style='color:blue;text-align:center;'>🔍 Predict Disease</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h3 style='color:black;'>Select Symptoms</h3>",
        unsafe_allow_html=True
    )

    selected_symptoms = st.multiselect(
        "Select Symptoms for accessibility",
        all_symptoms,
        label_visibility="collapsed"
    )

    if st.button("🧠 Predict Disease"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom")
        else:
            # Prepare input
            input_data = pd.DataFrame(
                np.zeros((1, len(all_symptoms))),
                columns=all_symptoms
            )
            for symptom in selected_symptoms:
                input_data[symptom] = 1

            # Predict
            prediction = model.predict(input_data)
            disease    = le.inverse_transform(prediction)[0]

            # Confidence (if model supports it)
            confidence = "N/A"
            if hasattr(model, "predict_proba"):
                proba      = model.predict_proba(input_data)[0]
                max_prob   = np.max(proba)
                confidence = f"{max_prob:.1%}"

            st.success(f"✅ Predicted Disease: **{disease}**")
            if confidence != "N/A":
                st.info(f"📊 Confidence: {confidence}")

            # ── Save result so app.py can log it to health records DB ──
            st.session_state["last_common_result"] = {
                "disease":    disease,
                "confidence": confidence,
                "symptoms":   selected_symptoms,      # list of selected symptoms
                "details": {
                    "total_symptoms_selected": len(selected_symptoms),
                    "symptoms_list":           ", ".join(selected_symptoms),
                }
            }
            # Reset saved flag so app.py logs this fresh result
            st.session_state["common_saved"] = False


# ---------- Run Page ----------
if __name__ == "__main__":
    common_disease_page()