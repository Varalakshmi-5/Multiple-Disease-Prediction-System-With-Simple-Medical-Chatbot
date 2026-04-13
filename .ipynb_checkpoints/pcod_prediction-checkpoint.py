import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ================= ENV FIX =================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.get_logger().setLevel("ERROR")

# ================= BACKGROUND =================
def prediction_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background: yellow;
        }
        .title-text {
            text-align: center;
            font-size: 38px;
            font-weight: 800;
            color: #2c3e50;
        }
        .subtitle-text {
            text-align: center;
            font-size: 16px;
            color: #555;
        }
        .result-box {
            padding: 22px;
            border-radius: 14px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ================= LOAD MODEL =================
@st.cache_resource
def load_pcod_model():
    return tf.keras.models.load_model("model/pcod_cnn_model.keras")

# ================= MAIN PAGE =================
def pcod_prediction_page():

    prediction_bg()
    model = load_pcod_model()

    IMG_SIZE = 224

    st.markdown("<div class='title-text'>🧬 PCOD Detection</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle-text'>Upload ultrasound image for PCOD prediction</div><br>",
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload Ultrasound Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        # ---------- IMAGE ----------
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # ---------- PREPROCESS ----------
        img = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype="float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ---------- PREDICTION ----------
        pred = model.predict(img_array, verbose=0)[0][0]
        confidence = pred * 100

        # ---------- DECISION ----------
        if pred >= 0.5:
            result = "Not Infected"
            st.markdown(
                f"""
                <div class='result-box' style='background:#fdecea;color:#c0392b;'>
                ✅ No PCOD Detected<br>
                Confidence: {confidence:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            result = "Infected"
            st.markdown(
                f"""
                <div class='result-box' style='background:#eafaf1;color:#1e8449;'>
                ⚠️ PCOD Detected<br>
                
                Confidence: {(100 - confidence):.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

        # ---------- DEBUG ----------
        with st.expander("🔍 Prediction Details"):
            st.write("Raw model output (sigmoid):", float(pred))
