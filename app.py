import streamlit as st
from streamlit_option_menu import option_menu
import base64
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Page config MUST be the very first Streamlit call ──────────────────────
st.set_page_config(
    page_title="Multiple Disease Prediction System with Simple Medical chatbot",
    page_icon="🩺",
    layout="wide"
)

# ── Auth gate ───────────────────────────────────────────────────────────────
from auth import auth_page, save_health_record, get_user_health_records, _build_report_html, log_report_download

if not auth_page():
    st.stop()

# ── Only authenticated regular users reach here ─────────────────────────────
from Diabetes_prediction       import diabetes_prediction_page
from common_disease_prediction import common_disease_page
from heart_prediction          import heart_prediction_page
from pcod_prediction           import pcod_prediction_page
from pneumonia_prediction      import pneumonia_prediction_page
from chatbot                   import medical_chatbot_page


# ── Helpers ──────────────────────────────────────────────────────────────────
def reset_bg_color():
    st.markdown("""
    <style>.stApp { background: #f5f7fa; }</style>
    """, unsafe_allow_html=True)

def set_sidebar_color(color):
    st.markdown(f"""
    <style>
    section[data-testid="stSidebar"] {{ background: {color}; }}
    section[data-testid="stSidebar"] * {{ color: white !important; }}
    </style>""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
user_info = st.session_state.get("user_info", {})

with st.sidebar:
    st.markdown(f"👤 **{user_info.get('name', 'User')}**")
    st.caption(user_info.get('email', ''))
    if st.button("🚪 Logout", key="main_logout"):
        for k in ["logged_in", "user_role", "user_info"]:
            st.session_state[k] = False if k == "logged_in" else None
        st.session_state.auth_page = "login"
        st.rerun()
    st.markdown("---")

    selected = option_menu(
        menu_title="Navigation",
        options=[
            "Home",
            "Diabetes Prediction",
            "Heart Disease Prediction",
            "Common Disease Prediction",
            "PCOD Prediction",
            "Pneumonia Prediction",
            "AI Chatbot",
            "My Health Report"
        ],
        icons=["house-fill","droplet-half","heart-pulse","activity","gender-female","lungs","robot","file-medical"],
        default_index=0,
        styles={
            "nav-link-selected": {
                "background-color": "#ffeb3b",
                "color": "#1b5e20",
                "font-weight": "bold",
            }
        }
    )


# ── Page routing ──────────────────────────────────────────────────────────────
if selected == "Home":
    set_sidebar_color("#E67E00")

    st.markdown("""
    <style>
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(24px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%       { opacity: 0.5; transform: scale(1.4); }
    }
    @keyframes bounce {
        0%, 100% { transform: translateX(0); }
        50%       { transform: translateX(-5px); }
    }
    .hp-root {
        min-height: 500px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 40%, #0c444c 100%);
        padding: 2.5rem 1.5rem;
        border-radius: 16px;
        position: relative;
        overflow: hidden;
    }
    .hp-root::before {
        content: '';
        position: absolute;
        top: -80px; left: -80px;
        width: 320px; height: 320px;
        background: rgba(59,130,246,0.12);
        border-radius: 50%;
        filter: blur(60px);
    }
    .hp-root::after {
        content: '';
        position: absolute;
        bottom: -60px; right: -60px;
        width: 280px; height: 280px;
        background: rgba(20,184,166,0.10);
        border-radius: 50%;
        filter: blur(60px);
    }
    .hp-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 800;
        color: #f0f9ff;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
    }
    .hp-sub {
        text-align: center;
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    .hp-pulse {
        display: inline-block;
        width: 10px; height: 10px;
        background: #34d399;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 1.8s ease-in-out infinite;
    }
    .card-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        position: relative;
        z-index: 1;
    }
    .hp-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.13);
        border-radius: 16px;
        padding: 22px 18px;
        cursor: pointer;
        transition: transform 0.28s cubic-bezier(.34,1.56,.64,1), background 0.2s, box-shadow 0.25s;
        position: relative;
        overflow: hidden;
        animation: fadeUp 0.5s ease both;
    }
    .hp-card::before {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(255,255,255,0.06), transparent);
        pointer-events: none;
    }
    .hp-card:hover {
        transform: translateY(-8px) scale(1.03);
        background: rgba(255,255,255,0.13);
        box-shadow: 0 16px 40px rgba(0,0,0,0.35);
    }
    .hp-card:nth-child(1) { animation-delay: .05s; }
    .hp-card:nth-child(2) { animation-delay: .10s; }
    .hp-card:nth-child(3) { animation-delay: .15s; }
    .hp-card:nth-child(4) { animation-delay: .20s; }
    .hp-card:nth-child(5) { animation-delay: .25s; }
    .hp-card:nth-child(6) { animation-delay: .30s; }
    .hp-icon {
        width: 44px; height: 44px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        margin-bottom: 14px;
    }
    .icon-red    { background: rgba(239,68,68,0.18); }
    .icon-pink   { background: rgba(236,72,153,0.18); }
    .icon-blue   { background: rgba(59,130,246,0.18); }
    .icon-purple { background: rgba(139,92,246,0.18); }
    .icon-cyan   { background: rgba(6,182,212,0.18); }
    .icon-green  { background: rgba(34,197,94,0.18); }
    .hp-card-title { font-size: 0.97rem; font-weight: 700; color: #f1f5f9; margin-bottom: 6px; }
    .hp-card-desc  { font-size: 0.82rem; color: #94a3b8; line-height: 1.5; }
    .hp-tag {
        display: inline-block;
        margin-top: 12px;
        font-size: 0.72rem;
        padding: 3px 10px;
        border-radius: 20px;
        font-weight: 600;
    }
    .tag-ml  { background: rgba(59,130,246,0.2);  color: #93c5fd; }
    .tag-dl  { background: rgba(139,92,246,0.2);  color: #c4b5fd; }
    .tag-nlp { background: rgba(34,197,94,0.2);   color: #86efac; }
    .hp-bottom {
        text-align: center;
        margin-top: 1.8rem;
        font-size: 0.85rem;
        color: #94a3b8;
        position: relative;
        z-index: 1;
    }
    .hp-arrow { display: inline-block; animation: bounce 1.5s ease-in-out infinite; }
    </style>

    <div class="hp-root">
        <div class="hp-title"><span class="hp-pulse"></span>Health Prediction System</div>
        <div class="hp-sub">AI-powered disease prediction using machine learning &amp; deep learning</div>
        <div class="card-grid">
            <div class="hp-card">
                <div class="hp-icon icon-red">🩸</div>
                <div class="hp-card-title">Diabetes Prediction</div>
                <div class="hp-card-desc">Predict risk using glucose, BMI, HbA1c and clinical markers.</div>
                <span class="hp-tag tag-ml">ML Model</span>
            </div>
            <div class="hp-card">
                <div class="hp-icon icon-pink">❤️</div>
                <div class="hp-card-title">Heart Disease Prediction</div>
                <div class="hp-card-desc">Assess cardiovascular risk using clinical parameters.</div>
                <span class="hp-tag tag-ml">ML Model</span>
            </div>
            <div class="hp-card">
                <div class="hp-icon icon-blue">🤖</div>
                <div class="hp-card-title">Common Disease Prediction</div>
                <div class="hp-card-desc">Multi-disease prediction from symptoms using ML.</div>
                <span class="hp-tag tag-ml">ML Model</span>
            </div>
            <div class="hp-card">
                <div class="hp-icon icon-purple">💊</div>
                <div class="hp-card-title">PCOD Prediction</div>
                <div class="hp-card-desc">Detect PCOD via ultrasound image analysis with CNN.</div>
                <span class="hp-tag tag-dl">Deep Learning</span>
            </div>
            <div class="hp-card">
                <div class="hp-icon icon-cyan">🫁</div>
                <div class="hp-card-title">Pneumonia Prediction</div>
                <div class="hp-card-desc">Diagnose pneumonia from chest X-rays using Grad-CAM.</div>
                <span class="hp-tag tag-dl">Deep Learning</span>
            </div>
            <div class="hp-card">
                <div class="hp-icon icon-green">💬</div>
                <div class="hp-card-title">AI Medical Chatbot</div>
                <div class="hp-card-desc">NLP-powered chatbot for 41 diseases across 13 features.</div>
                <span class="hp-tag tag-nlp">NLP</span>
            </div>
        </div>
        <div class="hp-bottom"><span class="hp-arrow">👈</span> Select a module from the sidebar to begin</div>
    </div>
    """, unsafe_allow_html=True)

elif selected == "Diabetes Prediction":
    reset_bg_color()
    set_sidebar_color("#FFC5C5")

    diabetes_prediction_page()

    result = st.session_state.get("last_diabetes_result")
    if result and not st.session_state.get("diabetes_saved"):
        save_health_record(
            user_info  = user_info,
            module     = "Diabetes Prediction",
            disease    = result.get("disease", "Unknown"),
            confidence = result.get("confidence", ""),
            symptoms   = result.get("symptoms", []),
            details    = result.get("details", {})
        )
        st.session_state["diabetes_saved"] = True

elif selected == "Heart Disease Prediction":
    reset_bg_color()
    set_sidebar_color("#FFB6C1")

    heart_prediction_page()

    result = st.session_state.get("last_heart_result")
    if result and not st.session_state.get("heart_saved"):
        save_health_record(
            user_info  = user_info,
            module     = "Heart Disease Prediction",
            disease    = result.get("disease", "Unknown"),
            confidence = result.get("confidence", ""),
            details    = result.get("details", {})
        )
        st.session_state["heart_saved"] = True

elif selected == "Common Disease Prediction":
    reset_bg_color()
    set_sidebar_color("blue")

    common_disease_page()

    result = st.session_state.get("last_common_result")
    if result and not st.session_state.get("common_saved"):
        save_health_record(
            user_info  = user_info,
            module     = "Common Disease Prediction",
            disease    = result.get("disease", "Unknown"),
            confidence = result.get("confidence", ""),
            symptoms   = result.get("symptoms", []),
            details    = result.get("details", {})
        )
        st.session_state["common_saved"] = True

elif selected == "PCOD Prediction":
    reset_bg_color()
    set_sidebar_color("#5FB3B3")

    pcod_prediction_page()

    result = st.session_state.get("last_pcod_result")
    if result and not st.session_state.get("pcod_saved"):
        save_health_record(
            user_info  = user_info,
            module     = "PCOD Prediction",
            disease    = result.get("disease", "Unknown"),
            confidence = result.get("confidence", ""),
            details    = result.get("details", {})
        )
        st.session_state["pcod_saved"] = True

elif selected == "Pneumonia Prediction":
    set_sidebar_color("#C6F4D6")

    pneumonia_prediction_page()

    result = st.session_state.get("last_pneumonia_result")
    if result and not st.session_state.get("pneumonia_saved"):
        save_health_record(
            user_info  = user_info,
            module     = "Pneumonia Prediction",
            disease    = result.get("disease", "Unknown"),
            confidence = result.get("confidence", ""),
            details    = result.get("details", {})
        )
        st.session_state["pneumonia_saved"] = True

elif selected == "AI Chatbot":
    set_sidebar_color("#4CAF50")
    medical_chatbot_page()

elif selected == "My Health Report":
    import json
    import pandas as pd
    reset_bg_color()
    set_sidebar_color("#1976d2")

    st.markdown("""
    <style>
    .report-header {
        font-size: 2rem; font-weight: 800; color: #1e3a5f;
        text-align: center; margin-bottom: 4px;
    }
    .report-sub {
        text-align: center; color: #64748b;
        font-size: 1rem; margin-bottom: 28px;
    }
    .rpt-badge {
        display: inline-block;
        padding: 5px 16px; margin: 4px;
        border-radius: 20px; font-size: 0.85rem; font-weight: 600;
    }
    .badge-normal  { background: lightcyan;   border:1px solid lightblue;  color: steelblue; }
    .badge-warning { background: lightyellow; border:1px solid gold;        color: darkgoldenrod; }
    .badge-danger  { background: mistyrose;   border:1px solid lightcoral;  color: crimson; }
    .info-card {
        background: white; border: 1.5px solid lightsteelblue;
        border-radius: 16px; padding: 20px 28px; margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(70,130,180,0.10);
        color: #1e293b;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='report-header'>📋 My Health Report</div>", unsafe_allow_html=True)
    st.markdown("<div class='report-sub'>All your predictions in one place</div>", unsafe_allow_html=True)

    user_id = user_info.get("id", 0)
    records = get_user_health_records(user_id)

    if not records:
        st.info("🩺 No predictions yet. Run any prediction module and your results will appear here.")
    else:
        diseases  = list(dict.fromkeys(r["disease"] for r in records))
        modules   = list(dict.fromkeys(r["module"]  for r in records))
        danger_kw = ["heart", "diabetes", "pneumonia", "cancer"]
        warn_kw   = ["pcod", "thyroid", "hypertension"]

        badges_html = ""
        for d in diseases:
            dl  = d.lower()
            cls = ("badge-danger"  if any(k in dl for k in danger_kw) else
                   "badge-warning" if any(k in dl for k in warn_kw)   else "badge-normal")
            badges_html += f"<span class='rpt-badge {cls}'>{d}</span>"

        st.markdown(f"""
        <div class='info-card'>
            <b style='font-size:1.1rem;color:#1e3a5f'>👤 {user_info.get('name','')}</b><br>
            <span style='color:slategray;font-size:0.9rem'>📧 {user_info.get('email','')}</span>
            <hr style='border:none;border-top:1px solid #e2e8f0;margin:12px 0'>
            <b style='color:#475569'>Total Predictions:</b> {len(records)} &nbsp;|&nbsp;
            <b style='color:#475569'>Modules Used:</b> {len(modules)}<br><br>
            <b style='color:#475569'>Conditions Detected:</b><br>
            <div style='margin-top:8px'>{badges_html}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📅 Prediction History")
        df_rec = pd.DataFrame(records)[["predicted_at", "module", "disease", "confidence", "symptoms"]]
        df_rec["symptoms"] = df_rec["symptoms"].apply(
            lambda s: ", ".join(json.loads(s or "[]")) or "—"
        )
        df_rec.columns = ["Date & Time", "Module", "Disease / Result", "Confidence", "Symptoms"]
        st.dataframe(df_rec, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 📥 Download Your Report")
        st.markdown(
            "Download your full health report as an HTML file. "
            "Open it in Chrome/Edge and press **Ctrl+P → Save as PDF** to print."
        )
        report_html = _build_report_html(user_info, records)
        if st.download_button(
            label="📥 Download My Health Report",
            data=report_html,
            file_name=f"my_health_report_{user_info.get('name','user').replace(' ','_')}.html",
            mime="text/html",
            key="user_download_report"
        ):
            log_report_download(
                downloaded_by=user_info.get("email",""),
                role="User",
                report_for=user_info.get("email","")
            )