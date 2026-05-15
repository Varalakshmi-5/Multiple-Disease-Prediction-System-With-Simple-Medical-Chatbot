import streamlit as st
import sqlite3
import hashlib
import smtplib
import random
import string
import time
import re
import json
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

import os

try:
    SMTP_EMAIL     = st.secrets["SMTP_EMAIL"]
    SMTP_APP_PASS  = st.secrets["SMTP_APP_PASS"]
    ADMIN_EMAIL    = st.secrets["ADMIN_EMAIL"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    SMTP_EMAIL     = os.getenv("SMTP_EMAIL")
    SMTP_APP_PASS  = os.getenv("SMTP_APP_PASS")
    ADMIN_EMAIL    = os.getenv("ADMIN_EMAIL")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
    
DB_PATH        = "users.db"


# ─────────────────────────────────────────────
#  DATABASE
# ─────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            email       TEXT    UNIQUE NOT NULL,
            password    TEXT    NOT NULL,
            created_at  TEXT    NOT NULL,
            last_login  TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS login_logs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            email      TEXT NOT NULL,
            role       TEXT NOT NULL,
            logged_at  TEXT NOT NULL,
            status     TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS health_records (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL,
            user_email    TEXT    NOT NULL,
            user_name     TEXT    NOT NULL,
            module        TEXT    NOT NULL,
            disease       TEXT    NOT NULL,
            confidence    TEXT,
            symptoms      TEXT,
            details       TEXT,
            predicted_at  TEXT    NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS report_downloads (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            downloaded_by   TEXT    NOT NULL,
            downloaded_role TEXT    NOT NULL,
            report_for      TEXT    NOT NULL,
            downloaded_at   TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(name: str, email: str, password: str) -> bool:
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO users (name, email, password, created_at) VALUES (?,?,?,?)",
            (name, email, hash_password(password), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def verify_user(email: str, password: str):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM users WHERE email=? AND password=?",
        (email, hash_password(password))
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE users SET last_login=? WHERE email=?",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), email)
        )
        conn.commit()
    conn.close()
    return dict(row) if row else None


def email_exists(email: str) -> bool:
    conn = get_db()
    row = conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    return row is not None


def log_login(email: str, role: str, status: str):
    conn = get_db()
    conn.execute(
        "INSERT INTO login_logs (email, role, logged_at, status) VALUES (?,?,?,?)",
        (email, role, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status)
    )
    conn.commit()
    conn.close()


def get_all_users():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, name, email, created_at, last_login FROM users ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_login_logs():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM login_logs ORDER BY id DESC LIMIT 100"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_user(user_id: int):
    conn = get_db()
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.execute("DELETE FROM health_records WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()


def log_report_download(downloaded_by: str, role: str, report_for: str):
    conn = get_db()
    conn.execute(
        "INSERT INTO report_downloads (downloaded_by, downloaded_role, report_for, downloaded_at) VALUES (?,?,?,?)",
        (downloaded_by, role, report_for, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()


def get_report_downloads():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM report_downloads ORDER BY downloaded_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_health_record(user_info: dict, module: str, disease: str,
                       confidence: str = "", symptoms: list = None, details: dict = None):
    conn = get_db()
    conn.execute("""
        INSERT INTO health_records
            (user_id, user_email, user_name, module, disease, confidence, symptoms, details, predicted_at)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (
        user_info.get("id", 0),
        user_info.get("email", ""),
        user_info.get("name", ""),
        module,
        disease,
        confidence,
        json.dumps(symptoms or []),
        json.dumps(details or {}),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()


def get_all_health_records():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM health_records ORDER BY predicted_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_user_health_records(user_id: int):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM health_records WHERE user_id=? ORDER BY predicted_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────
#  FORGOT PASSWORD — DB HELPER
# ─────────────────────────────────────────────
def reset_user_password(email: str, new_password: str) -> bool:
    conn = get_db()
    cur = conn.execute(
        "UPDATE users SET password=? WHERE email=?",
        (hash_password(new_password), email)
    )
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated


# ─────────────────────────────────────────────
#  OTP EMAIL
# ─────────────────────────────────────────────
def generate_otp(length=6) -> str:
    return ''.join(random.choices(string.digits, k=length))


def send_otp_email(to_email: str, otp: str, name: str = "") -> bool:
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Health System - Your OTP Code"
        msg["From"]    = SMTP_EMAIL
        msg["To"]      = to_email

        plain = f"Hi {name or to_email},\n\nYour OTP is: {otp}\n\nValid for 5 minutes.\n"
        msg.attach(MIMEText(plain, "plain"))

        html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background-color:#f0f4f8;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
         style="background-color:#f0f4f8;padding:40px 0;">
    <tr><td align="center">
      <table role="presentation" width="520" cellpadding="0" cellspacing="0"
             style="background-color:#ffffff;border-radius:16px;
                    box-shadow:0 4px 24px rgba(0,0,0,0.10);overflow:hidden;
                    font-family:Arial,Helvetica,sans-serif;">
        <tr>
          <td style="background-color:#1b5e20;padding:28px 36px;text-align:center;">
            <h1 style="margin:0;font-size:22px;color:#ffffff;font-weight:700;">
              🩺 Email Verification
            </h1>
          </td>
        </tr>
        <tr>
          <td style="padding:36px 40px;background-color:#ffffff;">
            <p style="margin:0 0 10px;font-size:16px;color:#1e293b;">
              Hi <strong style="color:#1b5e20;">{name or to_email}</strong>,
            </p>
            <p style="margin:0 0 28px;font-size:14px;color:#475569;line-height:1.6;">
              Use the OTP below to verify your email.
            </p>
            <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
              <tr>
                <td align="center"
                    style="background-color:#f0faf4;border:2px solid #b2dfcb;
                           border-radius:14px;padding:28px 20px;">
                  <p style="margin:0 0 8px;font-size:12px;color:#1b5e20;
                             letter-spacing:2px;text-transform:uppercase;font-weight:600;">
                    Your One-Time Password
                  </p>
                  <p style="margin:0;font-size:52px;font-weight:900;
                             letter-spacing:16px;color:#1b5e20;
                             font-family:Courier New,Courier,monospace;">
                    {otp}
                  </p>
                </td>
              </tr>
            </table>
            <p style="margin:28px 0 0;font-size:13px;color:#64748b;text-align:center;">
              Expires in <strong style="color:#1e293b;">5 minutes</strong>. Never share it.
            </p>
          </td>
        </tr>
        <tr>
          <td style="background-color:#f8fafc;padding:20px 40px;
                     border-top:1px solid #e2e8f0;text-align:center;">
            <p style="margin:0;font-size:12px;color:#94a3b8;">
              &copy; {datetime.now().year} Health Prediction System
            </p>
          </td>
        </tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SMTP_EMAIL, SMTP_APP_PASS)
            server.sendmail(SMTP_EMAIL, to_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Email error: {e}")
        return False


def is_otp_valid() -> bool:
    exp = st.session_state.get("otp_expiry")
    return exp is not None and datetime.now() < exp


# ─────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────
def valid_email(email: str) -> bool:
    return bool(re.match(r"^[\w\.-]+@[\w\.-]+\.\w{2,}$", email))


def valid_password(pw: str) -> bool:
    return (len(pw) >= 8
            and any(c.isupper() for c in pw)
            and any(c.isdigit() for c in pw)
            and any(c in "!@#$%^&*()_+-=[]{}|;':\",./<>?" for c in pw))


# ─────────────────────────────────────────────
#  AUTH CSS — light pistachio green background, dark green buttons
# ─────────────────────────────────────────────
AUTH_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #c8e6c9 0%, #a5d6a7 50%, #c8e6c9 100%) !important;
    min-height: 100vh;
}
[data-testid="stSidebar"]  { display: none !important; }
[data-testid="stHeader"]   { background: transparent !important; }
[data-testid="stToolbar"]  { display: none !important; }
footer                     { display: none !important; }

.auth-brand {
    text-align: center;
    padding: 28px 0 8px;
}
.auth-brand-icon {
    font-size: 3rem;
    display: block;
    margin-bottom: 6px;
}
.auth-brand-name {
    font-size: 1rem;
    font-weight: 700;
    color: #1b5e20;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.auth-brand-tagline {
    font-size: 0.78rem;
    color: #2e7d32;
    margin-top: 3px;
}

.auth-card {
    background: #ffffff;
    border-radius: 24px;
    padding: 36px 40px 32px;
    max-width: 460px;
    margin: 12px auto 40px;
    box-shadow: 0 8px 40px rgba(46,125,50,0.15), 0 2px 8px rgba(46,125,50,0.08);
    border: 1px solid #c8e6c9;
}
.auth-card-title {
    font-size: 1.7rem;
    font-weight: 800;
    color: #1b5e20;
    margin-bottom: 4px;
    text-align: center;
}
.auth-card-sub {
    font-size: 0.85rem;
    color: #4caf50;
    text-align: center;
    margin-bottom: 24px;
}

div[data-testid="stTextInput"] input,
div[data-baseweb="input"] input {
    background-color: #f1f8f1 !important;
    color: #1b5e20 !important;
    -webkit-text-fill-color: #1b5e20 !important;
    border: 1.5px solid #a5d6a7 !important;
    border-radius: 10px !important;
    font-size: 0.92rem !important;
    padding: 10px 14px !important;
    box-shadow: none !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-baseweb="input"] input:focus {
    border-color: #1b5e20 !important;
    background-color: #ffffff !important;
    box-shadow: 0 0 0 3px rgba(27,94,32,0.15) !important;
    outline: none !important;
}
input::placeholder {
    color: #81c784 !important;
    opacity: 1 !important;
}

div[data-testid="stTextInput"] label,
div[data-testid="stSelectbox"] label {
    color: #1b5e20 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    margin-bottom: 4px !important;
}

div[data-testid="stSelectbox"] > div > div {
    background-color: #f1f8f1 !important;
    color: #1b5e20 !important;
    border: 1.5px solid #a5d6a7 !important;
    border-radius: 10px !important;
    font-size: 0.92rem !important;
}
div[data-testid="stSelectbox"] svg { color: #1b5e20 !important; }

/* ── Dark green buttons ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #1b5e20, #2e7d32) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 13px 20px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 14px rgba(27,94,32,0.35) !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #145214, #1b5e20) !important;
    box-shadow: 0 6px 18px rgba(27,94,32,0.45) !important;
}
.stButton > button:active {
    transform: translateY(1px) !important;
}

.auth-divider {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 18px 0;
    color: #81c784;
    font-size: 0.78rem;
    font-weight: 500;
}
.auth-divider::before,
.auth-divider::after {
    content: '';
    flex: 1;
    border-top: 1px solid #c8e6c9;
}

.banner-error {
    background: #fff0f3;
    border: 1px solid #f4b8c8;
    border-left: 4px solid #e05478;
    border-radius: 10px;
    padding: 12px 16px;
    color: #7a1f38;
    font-size: 0.88rem;
    font-weight: 500;
    margin: 10px 0;
}
.banner-success {
    background: #f0faf4;
    border: 1px solid #b2dfcb;
    border-left: 4px solid #2e7d32;
    border-radius: 10px;
    padding: 12px 16px;
    color: #1b5e20;
    font-size: 0.88rem;
    font-weight: 500;
    margin: 10px 0;
}
.banner-info {
    background: #f5f0fa;
    border: 1px solid #d4b8e8;
    border-left: 4px solid #9b6dcc;
    border-radius: 10px;
    padding: 12px 16px;
    color: #4a2475;
    font-size: 0.88rem;
    font-weight: 500;
    margin: 10px 0;
}

.otp-timer {
    background: #f0faf4;
    border: 1px solid #b2dfcb;
    border-radius: 12px;
    padding: 14px 20px;
    text-align: center;
    color: #1b5e20;
    font-size: 0.9rem;
    font-weight: 600;
    margin: 12px 0;
}

.features-strip {
    display: flex;
    justify-content: center;
    gap: 28px;
    margin: 0 auto 20px;
    max-width: 460px;
}
.feature-pill {
    background: rgba(255,255,255,0.75);
    border: 1px solid #a5d6a7;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.75rem;
    color: #1b5e20;
    font-weight: 600;
    white-space: nowrap;
}
</style>
"""


# ─────────────────────────────────────────────
#  ADMIN CSS  ← sidebar changed to BLUE
# ─────────────────────────────────────────────
ADMIN_BG_COLOR      = "#f0faf4"
ADMIN_SIDEBAR_BG    = "#0d2f6e"          # ← deep blue (was #1b5e20)
ADMIN_SIDEBAR_TEXT  = "white"
ADMIN_ACCENT        = "#1b5e20"
ADMIN_CARD_BG       = "white"
ADMIN_CARD_BORDER   = "#b2dfcb"

ADMIN_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');

[data-testid="stAppViewContainer"] {{
    background-color: {ADMIN_BG_COLOR} !important;
    min-height: 100vh;
}}
[data-testid="stHeader"] {{ background: transparent !important; }}

section[data-testid="stSidebar"] {{
    background-color: {ADMIN_SIDEBAR_BG} !important;
    border-right: 2px solid #90b4e8 !important;
}}
section[data-testid="stSidebar"] * {{
    color: {ADMIN_SIDEBAR_TEXT} !important;
    font-family: 'DM Sans', sans-serif !important;
}}
section[data-testid="stSidebar"] .stRadio label {{
    background: rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    margin: 4px 0 !important;
    display: block !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
}}
section[data-testid="stSidebar"] .stRadio label:hover {{
    background: rgba(255,255,255,0.28) !important;
}}
section[data-testid="stSidebar"] button {{
    background: rgba(220,38,38,0.18) !important;
    border: 1px solid rgba(220,38,38,0.4) !important;
    color: lightsalmon !important;
    border-radius: 10px !important;
    width: 100% !important;
}}

.admin-page-title {{
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: {ADMIN_ACCENT};
    margin-bottom: 2px;
}}
.admin-page-sub {{
    color: #6b9e82;
    font-size: 0.85rem;
    font-family: 'DM Sans', sans-serif;
    margin-bottom: 20px;
}}

.stat-card {{
    background-color: {ADMIN_CARD_BG};
    border: 1.5px solid {ADMIN_CARD_BORDER};
    border-radius: 18px;
    padding: 24px 20px;
    text-align: center;
    font-family: 'DM Sans', sans-serif;
    box-shadow: 0 4px 16px rgba(27,94,32,0.10);
    transition: transform .2s, box-shadow .2s;
}}
.stat-card:hover {{ transform: translateY(-3px); box-shadow: 0 8px 24px rgba(27,94,32,0.18); }}
.stat-num   {{ font-size: 2.4rem; font-weight: 800; color: {ADMIN_ACCENT}; font-family:'Syne',sans-serif; }}
.stat-label {{ font-size: 0.82rem; color: #6b9e82; margin-top: 6px; }}
.stat-icon  {{ font-size: 1.8rem; margin-bottom: 8px; }}

.disease-badge {{
    display: inline-block;
    background: #e8f5e9; border: 1px solid #b2dfcb;
    color: #1b5e20; border-radius: 20px;
    padding: 4px 14px; font-size: 0.82rem; font-weight: 600; margin: 3px;
}}
.disease-badge.warning {{
    background: #fff8e1; border-color: #ffe082; color: #b8860b;
}}
.disease-badge.danger {{
    background: #fff0f3; border-color: #f4b8c8; color: #c0587a;
}}

.user-health-card {{
    background: white;
    border: 1.5px solid #b2dfcb;
    border-radius: 16px;
    padding: 20px 24px;
    margin: 12px 0;
    font-family: 'DM Sans', sans-serif;
    box-shadow: 0 2px 8px rgba(27,94,32,0.08);
    color: #1e293b;
}}
.user-health-name  {{ font-family:'Syne',sans-serif; font-size:1.05rem; color:#1b5e20; font-weight:700; }}
.user-health-email {{ color:#6b9e82; font-size:0.82rem; margin-bottom:10px; }}
.admin-divider {{ border:none; border-top:1.5px solid #d4ead9; margin:18px 0; }}
</style>
"""


# ─────────────────────────────────────────────
#  SESSION HELPERS
# ─────────────────────────────────────────────
def _init_session():
    defaults = {
        "auth_page":         "login",
        "pending_signup":    None,
        "otp_code":          None,
        "otp_expiry":        None,
        "logged_in":         False,
        "user_role":         None,
        "user_info":         None,
        # ── forgot-password state ──
        "fp_email":          None,   # email being reset
        "fp_otp_code":       None,
        "fp_otp_expiry":     None,
        "fp_otp_verified":   False,  # True once OTP confirmed
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────
#  LOGIN PAGE
# ─────────────────────────────────────────────
def _login_ui():
    st.markdown(AUTH_CSS, unsafe_allow_html=True)

    st.markdown("""
    <div class="auth-brand">
        <span class="auth-brand-icon">🩺</span>
        <div class="auth-brand-name">Multiple Disease Prediction System with Simple Medical Chatbot</div>
        <div class="auth-brand-tagline">AI-powered • Multi-disease • Medical Chatbot</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="features-strip">
        <span class="feature-pill">🔬 ML Models</span>
        <span class="feature-pill">🧠 Deep Learning</span>
        <span class="feature-pill">💬 AI Chatbot</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
    st.markdown("<div class='auth-card-title'>Welcome Back 👋</div>", unsafe_allow_html=True)
    st.markdown("<div class='auth-card-sub'>Sign in to your account to continue</div>", unsafe_allow_html=True)

    role     = st.selectbox("Login as", ["User", "Admin"], key="login_role_select")
    email    = st.text_input("Email Address", placeholder="you@example.com", key="login_email")
    password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_pass")

    if st.button("Sign In →", key="btn_login"):
        if not email or not password:
            st.markdown("<div class='banner-error'>⚠️ Please fill in all fields.</div>", unsafe_allow_html=True)
        elif role == "Admin":
            if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
                log_login(email, "Admin", "success")
                st.session_state.logged_in = True
                st.session_state.user_role = "Admin"
                st.session_state.user_info = {"name": "Administrator", "email": email}
                st.rerun()
            else:
                log_login(email, "Admin", "failed")
                st.markdown("<div class='banner-error'>❌ Invalid admin credentials.</div>", unsafe_allow_html=True)
        else:
            user = verify_user(email, password)
            if user:
                log_login(email, "User", "success")
                st.session_state.logged_in = True
                st.session_state.user_role = "User"
                st.session_state.user_info = user
                st.rerun()
            else:
                log_login(email, "User", "failed")
                st.markdown("<div class='banner-error'>❌ Invalid email or password.</div>", unsafe_allow_html=True)

    # ── Forgot Password link ──
    st.markdown(
        "<div style='text-align:right;margin-top:-6px;margin-bottom:4px'>"
        "<span style='font-size:0.82rem;color:#2e7d32;cursor:pointer' "
        "id='fp-link'>Forgot password?</span></div>",
        unsafe_allow_html=True
    )
    if st.button("🔑 Forgot Password?", key="goto_forgot"):
        st.session_state.auth_page = "forgot_password"
        st.rerun()

    st.markdown("<div class='auth-divider'>don't have an account?</div>", unsafe_allow_html=True)

    if st.button("Create New Account →", key="goto_signup"):
        st.session_state.auth_page = "signup"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIGNUP PAGE
# ─────────────────────────────────────────────
def _signup_ui():
    st.markdown(AUTH_CSS, unsafe_allow_html=True)

    st.markdown("""
    <div class="auth-brand">
        <span class="auth-brand-icon">🩺</span>
        <div class="auth-brand-name">Multiple Disease Prediction System With Simple Medical Chatbot</div>
        <div class="auth-brand-tagline">Create your free account</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
    st.markdown("<div class='auth-card-title'>Create Account ✨</div>", unsafe_allow_html=True)
    st.markdown("<div class='auth-card-sub'>Join to access AI-powered health predictions</div>", unsafe_allow_html=True)

    name     = st.text_input("Full Name",        placeholder="e.g. Varalakshmi/Rajesh",                     key="su_name")
    email    = st.text_input("Email Address",    placeholder="you@example.com",                         key="su_email")
    password = st.text_input("Password", type="password",
                              placeholder="Min 8 chars · 1 uppercase · 1 digit · 1 special",            key="su_pass")
    confirm  = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password", key="su_confirm")

    if st.button("Send OTP & Verify →", key="btn_signup"):
        error = None
        if not name or not email or not password or not confirm:
            error = "⚠️ All fields are required."
        elif not valid_email(email):
            error = "⚠️ Enter a valid email address."
        elif email_exists(email):
            error = "⚠️ An account with this email already exists."
        elif not valid_password(password):
            error = "⚠️ Password must be ≥8 chars with 1 uppercase, 1 digit, 1 special character."
        elif password != confirm:
            error = "⚠️ Passwords do not match."

        if error:
            st.markdown(f"<div class='banner-error'>{error}</div>", unsafe_allow_html=True)
        else:
            otp = generate_otp()
            with st.spinner("Sending OTP to your email…"):
                sent = send_otp_email(email, otp, name)
            if sent:
                st.session_state.otp_code       = otp
                st.session_state.otp_expiry     = datetime.now() + timedelta(minutes=5)
                st.session_state.pending_signup = {"name": name, "email": email, "password": password}
                st.session_state.auth_page      = "otp_verify"
                st.rerun()
            else:
                st.markdown("<div class='banner-error'>❌ Failed to send OTP. Check SMTP settings.</div>",
                            unsafe_allow_html=True)

    st.markdown("<div class='auth-divider'>already have an account?</div>", unsafe_allow_html=True)

    if st.button("← Back to Login", key="goto_login_from_signup"):
        st.session_state.auth_page = "login"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  OTP VERIFY PAGE  (signup)
# ─────────────────────────────────────────────
def _otp_ui():
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    pending = st.session_state.pending_signup or {}

    st.markdown("""
    <div class="auth-brand">
        <span class="auth-brand-icon">📧</span>
        <div class="auth-brand-name">Verify Your Email</div>
        <div class="auth-brand-tagline">One last step</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
    st.markdown("<div class='auth-card-title'>Check Your Email 📬</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='auth-card-sub'>We sent a 6-digit OTP to<br>"
        f"<strong style='color:#1b5e20'>{pending.get('email','')}</strong></div>",
        unsafe_allow_html=True
    )

    if st.session_state.otp_expiry:
        remaining  = max(0, int((st.session_state.otp_expiry - datetime.now()).total_seconds()))
        mins, secs = divmod(remaining, 60)
        color      = "#1b5e20" if remaining > 120 else "#b45309" if remaining > 60 else "#c0587a"
        st.markdown(
            f"<div class='otp-timer'>⏱️ OTP expires in "
            f"<span style='color:{color};font-size:1.1rem'>{mins:02d}:{secs:02d}</span></div>",
            unsafe_allow_html=True
        )

    otp_input = st.text_input(
        "Enter 6-digit OTP",
        placeholder="e.g. 4 8 2 1 3 7",
        max_chars=6,
        key="otp_input"
    )

    if st.button("Verify & Create Account ✓", key="btn_verify"):
        if not is_otp_valid():
            st.markdown("<div class='banner-error'>⏰ OTP expired. Please sign up again.</div>",
                        unsafe_allow_html=True)
            time.sleep(1.5)
            st.session_state.auth_page = "signup"
            st.rerun()
        elif otp_input.strip() != st.session_state.otp_code:
            st.markdown("<div class='banner-error'>❌ Incorrect OTP. Please try again.</div>",
                        unsafe_allow_html=True)
        else:
            success = create_user(pending["name"], pending["email"], pending["password"])
            if success:
                st.markdown("<div class='banner-success'>✅ Account created! Redirecting to login…</div>",
                            unsafe_allow_html=True)
                st.session_state.otp_code       = None
                st.session_state.otp_expiry     = None
                st.session_state.pending_signup = None
                time.sleep(1.5)
                st.session_state.auth_page = "login"
                st.rerun()
            else:
                st.markdown("<div class='banner-error'>❌ Account creation failed.</div>",
                            unsafe_allow_html=True)

    st.markdown("<div class='auth-divider'>or</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Resend OTP", key="btn_resend"):
            otp = generate_otp()
            with st.spinner("Resending…"):
                sent = send_otp_email(pending["email"], otp, pending.get("name", ""))
            if sent:
                st.session_state.otp_code   = otp
                st.session_state.otp_expiry = datetime.now() + timedelta(minutes=5)
                st.markdown("<div class='banner-success'>✅ OTP resent successfully!</div>",
                            unsafe_allow_html=True)
                st.rerun()
    with col2:
        if st.button("← Back to Signup", key="btn_back_otp"):
            st.session_state.auth_page = "signup"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  FORGOT PASSWORD — Step 1: enter email → send OTP
# ─────────────────────────────────────────────
def _forgot_password_ui():
    st.markdown(AUTH_CSS, unsafe_allow_html=True)

    st.markdown("""
    <div class="auth-brand">
        <span class="auth-brand-icon">🔑</span>
        <div class="auth-brand-name">Forgot Password</div>
        <div class="auth-brand-tagline">Reset your account password</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
    st.markdown("<div class='auth-card-title'>Reset Password 🔐</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='auth-card-sub'>Enter your registered email and we'll send you a reset OTP</div>",
        unsafe_allow_html=True
    )

    fp_email = st.text_input("Registered Email Address", placeholder="you@example.com", key="fp_email_input")

    if st.button("Send Reset OTP →", key="btn_fp_send_otp"):
        if not fp_email:
            st.markdown("<div class='banner-error'>⚠️ Please enter your email address.</div>",
                        unsafe_allow_html=True)
        elif not valid_email(fp_email):
            st.markdown("<div class='banner-error'>⚠️ Enter a valid email address.</div>",
                        unsafe_allow_html=True)
        elif not email_exists(fp_email):
            st.markdown(
                "<div class='banner-error'>❌ No account found with this email address.</div>",
                unsafe_allow_html=True
            )
        else:
            otp = generate_otp()
            with st.spinner("Sending OTP to your email…"):
                sent = send_otp_email(fp_email, otp, "")
            if sent:
                st.session_state.fp_email      = fp_email
                st.session_state.fp_otp_code   = otp
                st.session_state.fp_otp_expiry = datetime.now() + timedelta(minutes=5)
                st.session_state.fp_otp_verified = False
                st.session_state.auth_page     = "fp_otp_verify"
                st.rerun()
            else:
                st.markdown("<div class='banner-error'>❌ Failed to send OTP. Check SMTP settings.</div>",
                            unsafe_allow_html=True)

    st.markdown("<div class='auth-divider'>remembered your password?</div>", unsafe_allow_html=True)

    if st.button("← Back to Login", key="btn_fp_back_login"):
        st.session_state.auth_page = "login"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  FORGOT PASSWORD — Step 2: verify OTP
# ─────────────────────────────────────────────
def _fp_otp_verify_ui():
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    fp_email = st.session_state.get("fp_email", "")

    st.markdown("""
    <div class="auth-brand">
        <span class="auth-brand-icon">📧</span>
        <div class="auth-brand-name">Verify Reset OTP</div>
        <div class="auth-brand-tagline">Enter the OTP sent to your email</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
    st.markdown("<div class='auth-card-title'>Check Your Email 📬</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='auth-card-sub'>We sent a 6-digit OTP to<br>"
        f"<strong style='color:#1b5e20'>{fp_email}</strong></div>",
        unsafe_allow_html=True
    )

    exp = st.session_state.get("fp_otp_expiry")
    if exp:
        remaining  = max(0, int((exp - datetime.now()).total_seconds()))
        mins, secs = divmod(remaining, 60)
        color      = "#1b5e20" if remaining > 120 else "#b45309" if remaining > 60 else "#c0587a"
        st.markdown(
            f"<div class='otp-timer'>⏱️ OTP expires in "
            f"<span style='color:{color};font-size:1.1rem'>{mins:02d}:{secs:02d}</span></div>",
            unsafe_allow_html=True
        )

    otp_input = st.text_input("Enter 6-digit OTP", placeholder="e.g. 7 3 1 9 4 2",
                               max_chars=6, key="fp_otp_input")

    if st.button("Verify OTP ✓", key="btn_fp_verify_otp"):
        exp = st.session_state.get("fp_otp_expiry")
        if exp is None or datetime.now() >= exp:
            st.markdown("<div class='banner-error'>⏰ OTP expired. Please request a new one.</div>",
                        unsafe_allow_html=True)
            time.sleep(1.5)
            st.session_state.auth_page = "forgot_password"
            st.rerun()
        elif otp_input.strip() != st.session_state.fp_otp_code:
            st.markdown("<div class='banner-error'>❌ Incorrect OTP. Please try again.</div>",
                        unsafe_allow_html=True)
        else:
            st.session_state.fp_otp_verified = True
            st.session_state.auth_page = "fp_reset_password"
            st.rerun()

    st.markdown("<div class='auth-divider'>or</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Resend OTP", key="btn_fp_resend"):
            otp = generate_otp()
            with st.spinner("Resending…"):
                sent = send_otp_email(fp_email, otp, "")
            if sent:
                st.session_state.fp_otp_code   = otp
                st.session_state.fp_otp_expiry = datetime.now() + timedelta(minutes=5)
                st.markdown("<div class='banner-success'>✅ OTP resent successfully!</div>",
                            unsafe_allow_html=True)
                st.rerun()
    with col2:
        if st.button("← Back", key="btn_fp_otp_back"):
            st.session_state.auth_page = "forgot_password"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  FORGOT PASSWORD — Step 3: set new password
# ─────────────────────────────────────────────
def _fp_reset_password_ui():
    st.markdown(AUTH_CSS, unsafe_allow_html=True)

    # Guard: if OTP was not verified, redirect back
    if not st.session_state.get("fp_otp_verified"):
        st.session_state.auth_page = "forgot_password"
        st.rerun()

    fp_email = st.session_state.get("fp_email", "")

    st.markdown("""
    <div class="auth-brand">
        <span class="auth-brand-icon">🔒</span>
        <div class="auth-brand-name">Set New Password</div>
        <div class="auth-brand-tagline">Choose a strong new password</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
    st.markdown("<div class='auth-card-title'>New Password 🔑</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='auth-card-sub'>Setting new password for<br>"
        f"<strong style='color:#1b5e20'>{fp_email}</strong></div>",
        unsafe_allow_html=True
    )

    new_pass    = st.text_input("New Password", type="password",
                                placeholder="Min 8 chars · 1 uppercase · 1 digit · 1 special",
                                key="fp_new_pass")
    confirm_pass = st.text_input("Confirm New Password", type="password",
                                 placeholder="Re-enter new password",
                                 key="fp_confirm_pass")

    if st.button("Update Password ✓", key="btn_fp_update"):
        if not new_pass or not confirm_pass:
            st.markdown("<div class='banner-error'>⚠️ Both fields are required.</div>",
                        unsafe_allow_html=True)
        elif not valid_password(new_pass):
            st.markdown(
                "<div class='banner-error'>⚠️ Password must be ≥8 chars with 1 uppercase, "
                "1 digit, 1 special character.</div>",
                unsafe_allow_html=True
            )
        elif new_pass != confirm_pass:
            st.markdown("<div class='banner-error'>⚠️ Passwords do not match.</div>",
                        unsafe_allow_html=True)
        else:
            updated = reset_user_password(fp_email, new_pass)
            if updated:
                st.markdown(
                    "<div class='banner-success'>✅ Password updated successfully! Redirecting to login…</div>",
                    unsafe_allow_html=True
                )
                # Clear all forgot-password state
                st.session_state.fp_email        = None
                st.session_state.fp_otp_code     = None
                st.session_state.fp_otp_expiry   = None
                st.session_state.fp_otp_verified = False
                time.sleep(1.5)
                st.session_state.auth_page = "login"
                st.rerun()
            else:
                st.markdown("<div class='banner-error'>❌ Password update failed. Please try again.</div>",
                            unsafe_allow_html=True)

    st.markdown("<div class='auth-divider'>or</div>", unsafe_allow_html=True)

    if st.button("← Back to Login", key="btn_fp_reset_back"):
        st.session_state.fp_email        = None
        st.session_state.fp_otp_code     = None
        st.session_state.fp_otp_expiry   = None
        st.session_state.fp_otp_verified = False
        st.session_state.auth_page = "login"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  REPORT HTML
# ─────────────────────────────────────────────
def _build_report_html(user: dict, records: list) -> str:
    rows_html = ""
    for r in records:
        symptoms = ", ".join(json.loads(r.get("symptoms", "[]") or "[]")) or "—"
        rows_html += f"""
        <tr>
            <td>{r['predicted_at']}</td>
            <td><b>{r['module']}</b></td>
            <td style="color:#1b5e20;font-weight:700">{r['disease']}</td>
            <td>{r.get('confidence','—')}</td>
            <td>{symptoms}</td>
        </tr>"""

    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
    <title>Health Report — {user['name']}</title>
    <style>
        body {{ font-family:'Segoe UI',sans-serif; margin:40px; color:#1e293b; background:white; }}
        .header {{ display:flex; justify-content:space-between; align-items:center;
                   border-bottom:3px solid #1b5e20; padding-bottom:16px; margin-bottom:28px; }}
        .logo {{ font-size:1.5rem; font-weight:800; color:#1b5e20; }}
        .user-box {{ background:#f0faf4; border:1px solid #b2dfcb; border-radius:12px;
                     padding:18px 24px; margin-bottom:28px; }}
        .user-box h2 {{ margin:0 0 6px; color:#1b5e20; }}
        .user-box p  {{ margin:3px 0; color:#475569; font-size:0.92rem; }}
        table {{ width:100%; border-collapse:collapse; font-size:0.9rem; }}
        th {{ background:#1b5e20; color:white; padding:12px 14px; text-align:left; }}
        td {{ padding:10px 14px; border-bottom:1px solid #e8f5e9; color:#1e293b; }}
        tr:nth-child(even) td {{ background:#f7faf8; }}
        .footer {{ margin-top:40px; text-align:center; color:#6b9e82; font-size:0.8rem; }}
        @media print {{ button {{ display:none; }} }}
    </style>
    </head><body>
    <div class="header">
        <div class="logo">🩺 Multiple Disease Prediction System With Simple Medical Chatbot</div>
        <div style="color:#6b9e82">Health Report &nbsp;|&nbsp; {datetime.now().strftime('%d %b %Y')}</div>
    </div>
    <div class="user-box">
        <h2>{user['name']}</h2>
        <p>📧 {user['email']}</p>
        <p>📅 Member since: {user.get('created_at','—')}</p>
        <p>🕐 Last login: {user.get('last_login','—')}</p>
    </div>
    <h3 style="color:#1b5e20;margin-bottom:12px">
        📋 Prediction History ({len(records)} record{"s" if len(records)!=1 else ""})
    </h3>
    <table>
        <thead><tr>
            <th>Date & Time</th><th>Module</th>
            <th>Predicted Disease</th><th>Confidence</th><th>Symptoms</th>
        </tr></thead>
        <tbody>
            {rows_html if rows_html
             else '<tr><td colspan="5" style="text-align:center;color:#6b9e82">No records</td></tr>'}
        </tbody>
    </table>
    <div class="footer">
        ⚠️ For informational purposes only. Always consult a healthcare professional.<br>
        Generated on {datetime.now().strftime('%d %b %Y at %H:%M')}
    </div>
    </body></html>"""


# ─────────────────────────────────────────────
#  ADMIN DASHBOARD
# ─────────────────────────────────────────────
def admin_dashboard():
    st.markdown(ADMIN_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"""
        <div style='text-align:center;padding:16px 0 8px'>
            <div style='font-size:2.2rem'>🩺</div>
            <div style='font-size:1.1rem;font-weight:800;letter-spacing:1px;
                        color:{ADMIN_SIDEBAR_TEXT}'>ADMIN PANEL</div>
            <div style='font-size:0.75rem;color:rgba(255,255,255,0.6);margin-top:4px'>
                Health Prediction System
            </div>
        </div>
        <hr style='border:none;border-top:1px solid rgba(255,255,255,0.25);margin:10px 0'>
        """, unsafe_allow_html=True)

        admin_section = st.radio(
            "Navigate",
            ["📊 Dashboard", "👥 Users", "🏥 Health Records",
             "📈 Disease Statistics", "📋 Login Logs",
             "📥 Download Logs", "🗑️ Manage Users"],
            label_visibility="collapsed"
        )

        st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.25);margin:12px 0'>",
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style='font-size:0.8rem;color:rgba(255,255,255,0.7);padding:0 4px'>
            Logged in as<br>
            <b style='color:white'>{st.session_state.user_info['email']}</b>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪 Logout", key="admin_logout"):
            for k in ["logged_in", "user_role", "user_info"]:
                st.session_state[k] = False if k == "logged_in" else None
            st.session_state.auth_page = "login"
            st.rerun()

    users   = get_all_users()
    logs    = get_login_logs()
    records = get_all_health_records()
    today   = datetime.now().strftime("%Y-%m-%d")

    total_users   = len(users)
    logins_today  = sum(1 for l in logs if l["logged_at"].startswith(today) and l["status"] == "success")
    failed_today  = sum(1 for l in logs if l["logged_at"].startswith(today) and l["status"] == "failed")
    total_records = len(records)

    danger_kw = ["heart", "diabetes", "pneumonia", "cancer"]
    warn_kw   = ["pcod", "thyroid", "hypertension"]

    def make_badges(diseases):
        html = ""
        for d in diseases:
            dl  = d.lower()
            cls = ("danger" if any(k in dl for k in danger_kw)
                   else "warning" if any(k in dl for k in warn_kw) else "")
            html += f"<span class='disease-badge {cls}'>{d}</span>"
        return html

    if admin_section == "📊 Dashboard":
        st.markdown("<div class='admin-page-title'>📊 Dashboard Overview</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='admin-page-sub'>{datetime.now().strftime('%d %b %Y, %H:%M')}</div>",
                    unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        for col, icon, num, label in [
            (c1, "👥", total_users,   "Total Users"),
            (c2, "✅", logins_today,  "Logins Today"),
            (c3, "❌", failed_today,  "Failed Logins"),
            (c4, "🏥", total_records, "Health Records"),
        ]:
            col.markdown(f"""
            <div class='stat-card'>
                <div class='stat-icon'>{icon}</div>
                <div class='stat-num'>{num}</div>
                <div class='stat-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='admin-divider'>", unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("#### 🆕 Recent Signups")
            if users:
                df = pd.DataFrame(users[:8])[["name", "email", "created_at"]]
                df.columns = ["Name", "Email", "Joined"]
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No users yet.")

        with col_r:
            st.markdown("#### 🏥 Recent Predictions")
            if records:
                df = pd.DataFrame(records[:8])[["user_name", "module", "disease", "predicted_at"]]
                df.columns = ["User", "Module", "Disease", "Date"]
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No health records yet.")

        if records:
            st.markdown("<hr class='admin-divider'>", unsafe_allow_html=True)
            st.markdown("#### 👤 User — Disease Mapping")
            df_r    = pd.DataFrame(records)
            mapping = (df_r.groupby("user_name")["disease"]
                           .apply(lambda x: ", ".join(sorted(set(x))))
                           .reset_index())
            mapping.columns = ["User", "Diseases Predicted"]
            st.dataframe(mapping, use_container_width=True, hide_index=True)

    elif admin_section == "👥 Users":
        st.markdown("<div class='admin-page-title'>👥 Registered Users</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='admin-page-sub'>Total: {total_users} user(s)</div>", unsafe_allow_html=True)

        if users:
            df = pd.DataFrame(users)[["id", "name", "email", "created_at", "last_login"]]
            df.columns = ["ID", "Name", "Email", "Joined", "Last Login"]
            st.dataframe(df, use_container_width=True, hide_index=True)

            if records:
                st.markdown("<hr class='admin-divider'>", unsafe_allow_html=True)
                st.markdown("#### 🏥 Disease Summary per User")
                for u in users:
                    user_recs = [r for r in records if r["user_id"] == u["id"]]
                    if user_recs:
                        diseases = list(dict.fromkeys(r["disease"] for r in user_recs))
                        st.markdown(f"""
                        <div class='user-health-card'>
                            <div class='user-health-name'>👤 {u['name']}</div>
                            <div class='user-health-email'>📧 {u['email']}</div>
                            <div style='margin-top:8px'>{make_badges(diseases)}</div>
                        </div>""", unsafe_allow_html=True)

            st.markdown("<hr class='admin-divider'>", unsafe_allow_html=True)
            st.markdown("#### 🔍 Search User")
            q = st.text_input("Search by name or email", key="user_search", placeholder="Type to search…")
            if q:
                filtered = [u for u in users
                            if q.lower() in u["name"].lower() or q.lower() in u["email"].lower()]
                st.dataframe(pd.DataFrame(filtered) if filtered else pd.DataFrame(),
                             use_container_width=True, hide_index=True)
                if not filtered:
                    st.info("No users match that search.")
        else:
            st.info("No registered users.")

    elif admin_section == "🏥 Health Records":
        st.markdown("<div class='admin-page-title'>🏥 User Health Records</div>", unsafe_allow_html=True)
        st.markdown("<div class='admin-page-sub'>Every prediction linked to each user</div>",
                    unsafe_allow_html=True)

        if not users:
            st.info("No users registered yet.")
        else:
            opts = {"— Show All Users —": None,
                    **{f"{u['name']}  ({u['email']})": u for u in users}}
            sel_user = opts[st.selectbox("Filter by user", list(opts.keys()), key="hr_user_filter")]

            disp = records if sel_user is None else get_user_health_records(sel_user["id"])

            if disp:
                modules = ["All Modules"] + sorted(set(r["module"] for r in disp))
                sel_mod = st.selectbox("Filter by module", modules, key="hr_mod_filter")
                if sel_mod != "All Modules":
                    disp = [r for r in disp if r["module"] == sel_mod]

            st.markdown(f"**{len(disp)} record(s) found**")
            st.markdown("<hr class='admin-divider'>", unsafe_allow_html=True)

            if disp:
                from collections import defaultdict
                grouped = defaultdict(list)
                for r in disp:
                    grouped[r["user_email"]].append(r)

                for idx, (email, urecs) in enumerate(grouped.items()):
                    ur       = urecs[0]
                    diseases = list(dict.fromkeys(r["disease"] for r in urecs))
                    st.markdown(f"""
                    <div class='user-health-card'>
                        <div class='user-health-name'>👤 {ur['user_name']}</div>
                        <div class='user-health-email'>📧 {email} &nbsp;|&nbsp; {len(urecs)} prediction(s)</div>
                        <div style='margin:8px 0'>{make_badges(diseases)}</div>
                    </div>""", unsafe_allow_html=True)

                    with st.expander(f"📋 Full records — {ur['user_name']}"):
                        df_u = pd.DataFrame(urecs)[
                            ["predicted_at", "module", "disease", "confidence", "symptoms"]
                        ].copy()
                        df_u["symptoms"] = df_u["symptoms"].apply(
                            lambda s: ", ".join(json.loads(s or "[]")) or "—"
                        )
                        df_u.columns = ["Date", "Module", "Disease", "Confidence", "Symptoms"]
                        st.dataframe(df_u, use_container_width=True, hide_index=True)
                        st.markdown("---")
                        full_user = next(
                            (u for u in users if u["email"] == email),
                            {"name": ur["user_name"], "email": email,
                             "created_at": "—", "last_login": "—"}
                        )
                        report_html = _build_report_html(full_user, urecs)
                        st.markdown("**📄 Download Health Report:**")
                        if st.download_button(
                            label=f"📥 Download Report for {ur['user_name']}",
                            data=report_html,
                            file_name=f"health_report_{ur['user_name'].replace(' ', '_')}.html",
                            mime="text/html",
                            key=f"dl_report_{idx}_{email.replace('@','_').replace('.','_')}"
                        ):
                            log_report_download(
                                downloaded_by=st.session_state.user_info["email"],
                                role="Admin",
                                report_for=email
                            )
                        st.caption("💡 After downloading, open in Chrome/Edge → Ctrl+P → Save as PDF.")
            else:
                st.info("No health records match the filter.")

    elif admin_section == "📈 Disease Statistics":
        st.markdown("<div class='admin-page-title'>📈 Disease Statistics</div>", unsafe_allow_html=True)
        st.markdown("<div class='admin-page-sub'>Insights across all predictions in the system</div>",
                    unsafe_allow_html=True)

        if not records:
            st.info("No prediction records yet.")
        else:
            df_r = pd.DataFrame(records)
            st.markdown("<hr class='admin-divider'>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🦠 Top Predicted Diseases")
                top_d = df_r["disease"].value_counts().head(10).reset_index()
                top_d.columns = ["Disease", "Count"]
                st.bar_chart(top_d.set_index("Disease"))

            with col2:
                st.markdown("#### 🔬 Predictions by Module")
                mod_c = df_r["module"].value_counts().reset_index()
                mod_c.columns = ["Module", "Count"]
                st.bar_chart(mod_c.set_index("Module"))

            st.markdown("<hr class='admin-divider'>", unsafe_allow_html=True)
            st.markdown("#### 👤 User — Disease Mapping")
            mapping = (df_r.groupby(["user_name", "user_email"])["disease"]
                           .apply(lambda x: ", ".join(sorted(set(x))))
                           .reset_index())
            mapping.columns = ["User", "Email", "Diseases Predicted"]
            st.dataframe(mapping, use_container_width=True, hide_index=True)

            st.markdown("<hr class='admin-divider'>", unsafe_allow_html=True)
            st.markdown("#### 📅 Predictions Over Time")
            df_r["date"] = pd.to_datetime(df_r["predicted_at"]).dt.date
            daily = df_r.groupby("date").size().reset_index(name="Predictions").set_index("date")
            st.line_chart(daily)

            st.markdown("<hr class='admin-divider'>", unsafe_allow_html=True)
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("#### 📊 Disease Count Table")
                stats = df_r["disease"].value_counts().reset_index()
                stats.columns = ["Disease", "Count"]
                stats["% Share"] = (stats["Count"] / stats["Count"].sum() * 100).round(1).astype(str) + "%"
                st.dataframe(stats, use_container_width=True, hide_index=True)
            with col4:
                st.markdown("#### 🏆 Most Active Users")
                active = df_r["user_name"].value_counts().reset_index()
                active.columns = ["User", "Total Predictions"]
                st.dataframe(active, use_container_width=True, hide_index=True)

    elif admin_section == "📋 Login Logs":
        st.markdown("<div class='admin-page-title'>📋 Login Activity</div>", unsafe_allow_html=True)
        st.markdown("<div class='admin-page-sub'>Last 100 login attempts</div>", unsafe_allow_html=True)

        if logs:
            df = pd.DataFrame(logs)[["email", "role", "logged_at", "status"]]
            df.columns = ["Email", "Role", "Timestamp", "Status"]
            st.dataframe(
                df.style.map(
                    lambda v: "color:green;font-weight:600" if v == "success" else "color:crimson;font-weight:600",
                    subset=["Status"]
                ),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("No login activity recorded.")

    elif admin_section == "📥 Download Logs":
        st.markdown("<div class='admin-page-title'>📥 Report Download Logs</div>", unsafe_allow_html=True)
        st.markdown("<div class='admin-page-sub'>Track every report downloaded by users and admin</div>",
                    unsafe_allow_html=True)

        dl_logs = get_report_downloads()

        if not dl_logs:
            st.info("No reports have been downloaded yet.")
        else:
            total_dl = len(dl_logs)
            user_dl  = sum(1 for d in dl_logs if d["downloaded_role"] == "User")
            admin_dl = sum(1 for d in dl_logs if d["downloaded_role"] == "Admin")
            today_dl = sum(1 for d in dl_logs if d["downloaded_at"].startswith(today))

            c1, c2, c3, c4 = st.columns(4)
            for col, icon, num, label in [
                (c1, "📥", total_dl, "Total Downloads"),
                (c2, "👤", user_dl,  "By Users"),
                (c3, "🛡️", admin_dl, "By Admin"),
                (c4, "📅", today_dl, "Today"),
            ]:
                col.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-icon'>{icon}</div>
                    <div class='stat-num'>{num}</div>
                    <div class='stat-label'>{label}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<hr class='admin-divider'>", unsafe_allow_html=True)

            df_dl = pd.DataFrame(dl_logs)[["downloaded_by", "downloaded_role", "report_for", "downloaded_at"]]
            df_dl.columns = ["Downloaded By", "Role", "Report For (User)", "Date & Time"]

            def color_role(val):
                return ("color:#1b5e20;font-weight:600" if val == "User"
                        else "color:#c0587a;font-weight:600")

            st.dataframe(
                df_dl.style.map(color_role, subset=["Role"]),
                use_container_width=True, hide_index=True
            )

    elif admin_section == "🗑️ Manage Users":
        st.markdown("<div class='admin-page-title'>🗑️ Manage Users</div>", unsafe_allow_html=True)
        st.markdown("<div class='admin-page-sub'>Delete a user and all their records permanently.</div>",
                    unsafe_allow_html=True)

        if users:
            options   = {f"{u['name']}  ({u['email']})": u["id"] for u in users}
            sel_label = st.selectbox("Select user to delete", list(options.keys()))
            sel_id    = options[sel_label]
            user_recs = get_user_health_records(sel_id)

            st.markdown(f"""
            <div class='user-health-card'>
                <b style='color:#c0587a'>⚠️ This will permanently delete:</b><br>
                • User account: <b>{sel_label}</b><br>
                • All <b>{len(user_recs)}</b> associated health record(s)
            </div>""", unsafe_allow_html=True)

            if st.checkbox("I understand this action cannot be undone."):
                if st.button("🗑️ Delete User & Records", key="btn_delete_user"):
                    delete_user(sel_id)
                    st.success("✅ Deleted successfully.")
                    time.sleep(1)
                    st.rerun()
        else:
            st.info("No users to manage.")


# ─────────────────────────────────────────────
#  PUBLIC ENTRY POINT
# ─────────────────────────────────────────────
def auth_page() -> bool:
    init_db()
    _init_session()

    if st.session_state.logged_in:
        if st.session_state.user_role == "Admin":
            admin_dashboard()
            return False
        return True

    page = st.session_state.auth_page
    if page == "login":               _login_ui()
    elif page == "signup":            _signup_ui()
    elif page == "otp_verify":        _otp_ui()
    elif page == "forgot_password":   _forgot_password_ui()
    elif page == "fp_otp_verify":     _fp_otp_verify_ui()
    elif page == "fp_reset_password": _fp_reset_password_ui()
    return False
