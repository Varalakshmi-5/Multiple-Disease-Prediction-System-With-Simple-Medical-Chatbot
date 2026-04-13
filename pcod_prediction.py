import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image


# ── Load model once ───────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = 'best_pcod_model.keras'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None


def pcod_prediction_page():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 40%, #e0f2f1 100%);
    }
    .pcod-title {
        text-align: center;
        font-size: 38px;
        font-weight: 800;
        color: #c0392b;
        margin-bottom: 5px;
    }
    .pcod-subtitle {
        text-align: center;
        font-size: 16px;
        color: #555;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 22px;
        font-weight: 700;
        margin-top: 20px;
    }
    .result-infected {
        background-color: #fdecea;
        color: #c0392b;
        border: 2px solid #e74c3c;
    }
    .result-normal {
        background-color: #eafaf1;
        color: #1e8449;
        border: 2px solid #27ae60;
    }
    .confidence-box {
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 10px;
        padding: 15px;
        margin-top: 15px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Title ─────────────────────────────────────────────
    st.markdown("<div class='pcod-title'>💊 PCOD Prediction</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='pcod-subtitle'>Upload an ovarian ultrasound image to predict PCOD</div>",
        unsafe_allow_html=True
    )

    # ── Load model ────────────────────────────────────────
    model = load_model()
    if model is None:
        st.error("⚠️ Model not found! Make sure 'best_pcod_model.keras' is in the project folder.")
        return

    CLASS_NAMES = ['infected', 'notinfected']

    # ── Layout ────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📤 Upload Ultrasound Image")
        uploaded_file = st.file_uploader(
            "Choose an ultrasound image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an ovarian ultrasound image for PCOD prediction"
        )

        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption='Uploaded Image', use_container_width=True)

    with col2:
        st.markdown("### 🔍 Prediction Result")

        if uploaded_file is not None:
            # ── Preprocess ────────────────────────────────
            img_resized = img.resize((128, 128))
            img_array   = np.array(img_resized) / 255.0
            img_batch   = np.expand_dims(img_array, axis=0)

            # ── Predict ───────────────────────────────────
            with st.spinner('Analyzing image...'):
                probs      = model.predict(img_batch, verbose=0)[0]
                pred_idx   = np.argmax(probs)
                pred_class = CLASS_NAMES[pred_idx]
                confidence = probs[pred_idx] * 100

            # ── Result display ────────────────────────────
            if pred_class == 'infected':
                disease = "PCOD Detected"
                st.markdown("""
                <div class='result-box result-infected'>
                    🔴 PCOD Detected
                </div>
                """, unsafe_allow_html=True)
            else:
                disease = "No PCOD (Normal)"
                st.markdown("""
                <div class='result-box result-normal'>
                    🟢 Normal (No PCOD)
                </div>
                """, unsafe_allow_html=True)

            # ── Confidence scores ─────────────────────────
            st.markdown("<div class='confidence-box'>", unsafe_allow_html=True)
            st.markdown("#### 📊 Confidence Scores")
            for i, cls in enumerate(CLASS_NAMES):
                score = probs[i] * 100
                label = '🔴 Infected' if cls == 'infected' else '🟢 Normal'
                st.progress(int(score), text=f"{label}: {score:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Medical advice ────────────────────────────
            st.markdown("---")
            if pred_class == 'infected':
                st.warning("""
                ⚠️ **Important Notice:**
                This result suggests possible PCOD.
                Please consult a gynecologist for proper diagnosis and treatment.
                """)
            else:
                st.success("""
                ✅ **Good News:**
                No signs of PCOD detected.
                Continue regular health checkups.
                """)

            # ── Save result so app.py can log it to health records DB ──
            st.session_state["last_pcod_result"] = {
                "disease":    disease,
                "confidence": f"{confidence:.1f}%",
                "symptoms":   [],
                "details": {
                    "image_file":       uploaded_file.name,
                    "predicted_class":  pred_class,
                    "infected_score":   f"{probs[0] * 100:.1f}%",
                    "normal_score":     f"{probs[1] * 100:.1f}%",
                }
            }
            # Reset saved flag so app.py logs this fresh result
            st.session_state["pcod_saved"] = False

        else:
            st.info("👈 Please upload an ultrasound image to get prediction.")