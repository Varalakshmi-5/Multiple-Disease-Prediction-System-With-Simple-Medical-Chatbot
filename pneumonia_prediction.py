import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import os

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "pneumonia_model.keras")
INDICES_PATH = os.path.join(BASE_DIR, "class_indices.json")
IMG_SIZE     = (224, 224)


# ─────────────────────────────────────────────
# LOAD FUNCTIONS
# ─────────────────────────────────────────────

def load_pneumonia_model():
    if not Path(MODEL_PATH).exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_indices():
    if not Path(INDICES_PATH).exists():
        return {"NORMAL": 0, "PNEUMONIA": 1}
    with open(INDICES_PATH) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# PROCESSING FUNCTIONS
# ─────────────────────────────────────────────

def preprocess_image(uploaded_file):
    img_pil   = Image.open(uploaded_file).convert("RGB")
    img       = img_pil.resize(IMG_SIZE)
    arr       = np.array(img) / 255.0
    arr_batch = np.expand_dims(arr, axis=0).astype(np.float32)
    return img_pil, arr_batch


def predict(model, arr_batch, class_indices, threshold=0.5):
    prob          = float(model.predict(arr_batch, verbose=0)[0][0])
    pneumonia_idx = class_indices.get("PNEUMONIA", 1)
    if pneumonia_idx == 1:
        label      = "PNEUMONIA" if prob >= threshold else "NORMAL"
        confidence = prob if label == "PNEUMONIA" else 1 - prob
    else:
        label      = "NORMAL" if prob >= threshold else "PNEUMONIA"
        confidence = prob if label == "NORMAL" else 1 - prob
    return label, confidence, prob


def get_risk_level(prob, class_indices):
    pneumonia_idx  = class_indices.get("PNEUMONIA", 1)
    pneumonia_prob = prob if pneumonia_idx == 1 else 1 - prob
    if pneumonia_prob > 0.75:
        return "High", "#dc2626"
    elif pneumonia_prob > 0.4:
        return "Medium", "#d97706"
    else:
        return "Low", "#16a34a"


def make_gradcam(model, arr_batch):
    try:
        mobilenet  = model.get_layer("mobilenetv2_1.00_224")
        conv_layer = mobilenet.get_layer("Conv_1")

        mobilenet_conv_model = tf.keras.models.Model(
            inputs  = mobilenet.input,
            outputs = conv_layer.output
        )

        img_tensor = tf.constant(arr_batch)

        with tf.GradientTape() as tape:
            conv_outputs = mobilenet_conv_model(img_tensor)
            tape.watch(conv_outputs)

            x     = conv_outputs
            found = False
            for layer in mobilenet.layers:
                if layer.name == "Conv_1":
                    found = True
                    continue
                if found:
                    x = layer(x)

            for layer in model.layers[2:]:
                x = layer(x)

            loss = x[:, 0]

        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            return None

        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam     = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)
        cam     = tf.maximum(cam, 0)
        cam     = cam / (tf.math.reduce_max(cam) + 1e-8)
        cam     = cam.numpy()

        cam_resized = cv2.resize(cam, IMG_SIZE)
        return cam_resized

    except Exception:
        return None


def overlay_gradcam(original_pil, heatmap, alpha=0.4):
    img             = np.array(original_pil.resize(IMG_SIZE))
    heatmap_uint8   = np.uint8(255 * heatmap)
    colormap        = plt.get_cmap("jet")
    heatmap_colored = np.uint8(255 * colormap(heatmap_uint8)[:, :, :3])
    overlaid        = np.uint8(img * (1 - alpha) + heatmap_colored * alpha)
    return Image.fromarray(overlaid)


# ─────────────────────────────────────────────
# UI FUNCTIONS
# ─────────────────────────────────────────────

def set_background():
    st.markdown("""
    <style>
    .stApp {
        background-color:#FFC5C5 ;
    }
    </style>
    """, unsafe_allow_html=True)


def show_result_banner(label, confidence):
    css_class = "pneu-pneumonia" if label == "PNEUMONIA" else "pneu-normal"
    icon      = "🔴" if label == "PNEUMONIA" else "🟢"
    st.markdown(
        f'<div class="{css_class}">'
        f'{icon} Prediction: <strong>{label}</strong> &nbsp;|&nbsp; '
        f'Confidence: <strong>{confidence:.1%}</strong>'
        f'</div>',
        unsafe_allow_html=True
    )


def show_metric_cards(label, confidence, prob, class_indices):
    risk_level, risk_color = get_risk_level(prob, class_indices)
    label_color = "#dc2626" if label == "PNEUMONIA" else "#16a34a"
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="pneu-metric-card">
          <div class="pneu-metric-value" style="color:{label_color}">{label}</div>
          <div class="pneu-metric-label">Diagnosis</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="pneu-metric-card">
          <div class="pneu-metric-value">{confidence:.1%}</div>
          <div class="pneu-metric-label">Confidence</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="pneu-metric-card">
          <div class="pneu-metric-value" style="color:{risk_color}">{risk_level}</div>
          <div class="pneu-metric-label">Pneumonia Risk</div>
        </div>""", unsafe_allow_html=True)


def show_images(orig_img, cam_img, show_gradcam):
    if show_gradcam and cam_img is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original X-ray**")
            st.image(orig_img, use_container_width=True)
        with col2:
            st.markdown("**Grad-CAM — Model Focus Areas**")
            st.image(cam_img, use_container_width=True)
            st.caption("🔴 Red/yellow = regions influencing the prediction")
    else:
        st.image(orig_img, caption="Uploaded X-ray", width=350)


def show_probability_bars(prob, class_indices):
    pneumonia_idx  = class_indices.get("PNEUMONIA", 1)
    pneumonia_prob = prob if pneumonia_idx == 1 else 1 - prob
    normal_prob    = 1 - pneumonia_prob
    st.markdown("**Probability Breakdown**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("NORMAL probability", f"{normal_prob:.1%}")
        st.progress(float(normal_prob))
    with col2:
        st.metric("PNEUMONIA probability", f"{pneumonia_prob:.1%}")
        st.progress(float(pneumonia_prob))


def show_advice(label):
    if label == "PNEUMONIA":
        st.warning("""
        **Note:** Signs of pneumonia detected.
        Please consult a qualified radiologist or physician for a proper diagnosis.
        This tool is for educational/research purposes only.
        """)
    else:
        st.success("""
        **Note:** No obvious signs of pneumonia detected.
        Always consult a healthcare professional for clinical decisions.
        """)


def show_placeholder():
    st.info("👆 Upload a chest X-ray image to get started.")


# ─────────────────────────────────────────────
# MAIN PAGE FUNCTION
# ─────────────────────────────────────────────

def pneumonia_prediction_page():

    set_background()

    st.markdown("""
    <style>
      .pneu-pneumonia {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-size: 1.1rem;
        font-weight: 600;
        background-color: pink;
        color: #991b1b;
        border-left: 5px solid #ef4444;
      }
      .pneu-normal {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-size: 1.1rem;
        font-weight: 600;
        background-color: #FFC5C5;
        color: #166534;
        border-left: 5px solid #22c55e;
      }
      .pneu-metric-card {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid #bae6fd;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 1rem;
      }
      .pneu-metric-value { font-size: 1.6rem; font-weight: 700; }
      .pneu-metric-label { font-size: 0.8rem; color: #0369a1; margin-top: 4px; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🫁 Pneumonia Detection from Chest X-Ray")
    st.markdown("Upload a chest X-ray and the model will predict **NORMAL** or **PNEUMONIA**.")
    st.markdown("---")

    model         = load_pneumonia_model()
    class_indices = load_class_indices()

    if model is None:
        st.error(f"Model not found at `{MODEL_PATH}`. Please complete training first.")
        return

    show_gradcam = st.checkbox("Show Grad-CAM heatmap", value=True)

    uploaded = st.file_uploader(
        "Upload a chest X-ray image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        st.markdown("---")
        with st.spinner("Analysing X-ray..."):
            orig_img, arr_batch     = preprocess_image(uploaded)
            label, confidence, prob = predict(model, arr_batch, class_indices)
            heatmap                 = make_gradcam(model, arr_batch)
            cam_img                 = overlay_gradcam(orig_img, heatmap) if heatmap is not None else None

        show_result_banner(label, confidence)
        show_metric_cards(label, confidence, prob, class_indices)
        st.markdown("<br>", unsafe_allow_html=True)
        show_images(orig_img, cam_img, show_gradcam)
        st.markdown("---")
        show_probability_bars(prob, class_indices)
        st.markdown("---")
        show_advice(label)

        pneumonia_idx  = class_indices.get("PNEUMONIA", 1)
        pneumonia_prob = prob if pneumonia_idx == 1 else 1 - prob
        risk_level, _  = get_risk_level(prob, class_indices)

        st.session_state["last_pneumonia_result"] = {
            "disease":    label,
            "confidence": f"{confidence:.1%}",
            "symptoms":   [],
            "details": {
                "image_file":      uploaded.name,
                "pneumonia_prob":  f"{pneumonia_prob:.1%}",
                "normal_prob":     f"{1 - pneumonia_prob:.1%}",
                "risk_level":      risk_level,
                "gradcam_shown":   show_gradcam,
            }
        }
        st.session_state["pneumonia_saved"] = False

    else:
        show_placeholder()
