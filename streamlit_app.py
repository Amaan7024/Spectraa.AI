import streamlit as st

import streamlit as st

st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFF176;
    }
    </style>
    """,
    unsafe_allow_html=True
)



import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model_core import preprocess_spectrum, train_model, predict_spectrum

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
from datetime import datetime
from datetime import datetime
import pytz

def generate_pdf_report(present_count, absent_count, test_fig, pred, conf):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2, height - 50, "Spectral.AI Analysis Report")

    # Timestamp
    ist = pytz.timezone("Asia/Kolkata")
    timestamp = datetime.now(pytz.utc).astimezone(ist).strftime("%Y-%m-%d %H:%M:%S")
    c.setFont("Helvetica", 10)
    c.drawRightString(width - 40, height - 70, f"Generated on: {timestamp}")

    # Data summary
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 120, "Uploaded Data Summary:")
    c.setFont("Helvetica", 12)
    c.drawString(70, height - 140, f"Target Present Files: {present_count}")
    c.drawString(70, height - 160, f"Target Absent Files:  {absent_count}")

    # Example counts (you already have present_files and absent_files lists)
    present_count = len(present_files) if present_files else 0
    absent_count = len(absent_files) if absent_files else 0

         # Pie chart + Spectrum plot (side by side, properly sized)
    if test_fig is not None and (present_count + absent_count > 0):
        # --- Pie Chart ---
        pie_buf = io.BytesIO()
        labels = ['Target Present', 'Target Absent']
        sizes = [present_count, absent_count]
        colors = ['#66bb6a', '#ef5350']

        fig_pie, ax_pie = plt.subplots(figsize=(3,3))
        ax_pie.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                   colors=colors, shadow=True, textprops={'color':"black"})
        ax_pie.axis('equal')

        fig_pie.savefig(pie_buf, format="png", bbox_inches="tight")
        plt.close(fig_pie)
        pie_buf.seek(0)
        pie_img = ImageReader(pie_buf)

        # --- Spectrum Plot ---
        spec_buf = io.BytesIO()
        test_fig.set_size_inches(3.5,3)   # shrink spectrum plot
        test_fig.savefig(spec_buf, format="png", bbox_inches="tight")
        spec_buf.seek(0)
        spec_img = ImageReader(spec_buf)

        # --- Place them side by side (fit page width) ---
        img_w, img_h = 220, 180   # resized
        margin_left = 60
        spacing = 40

        c.drawImage(pie_img, margin_left, height - 450, width=img_w, height=img_h, mask='auto')
        c.drawImage(spec_img, margin_left + img_w + spacing, height - 450, width=img_w, height=img_h, mask='auto')

        # Titles above images
        c.setFont("Helvetica-Bold", 12)
        c.drawCentredString(margin_left + img_w/2, height - 250, "Data Distribution")
        c.drawCentredString(margin_left + img_w + spacing + img_w/2, height - 250, "Test Spectrum")



               # --- Prediction Section just below the plots ---
    if pred == 1:
        result_text = "MODEL PREDICTION: TARGET PRESENT ✅"
        c.setFillColorRGB(0, 0.6, 0)   # green
    else:
        result_text = "MODEL PREDICTION: TARGET ABSENT ❌"
        c.setFillColorRGB(0.8, 0, 0)   # red

    c.setFont("Helvetica-Bold", 16)

    # Place just below spectrum plot area
    y_position = height - 520  
    c.drawCentredString(width / 2, y_position, result_text)

    # Confidence (always black)
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width / 2, y_position - 20, f"Confidence: {conf:.2%}")




    c.save()
    buffer.seek(0)
    return buffer
    


import streamlit as st



# Global modern background with blur and polish
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
        color: #f0f0f0;
    }
    .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        margin: auto;
    }
    h1, h2, h3 {
        color: #ffffff;
        text-align: center;
    }
    .stButton>button {
        background-color: #3a47d5;
        color: white;
        font-weight: bold;
        border-radius: 30px;
        padding: 10px 24px;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2936a1;
        transform: scale(1.02);
    }
    </style>
    """,
    unsafe_allow_html=True
)


import base64
from PIL import Image

# Load the banner image
banner = Image.open("INST.LDMD.png")  

# Display as full-width header banner
st.markdown(
    f"""
    <div style="text-align:center;">
        <img src="data:image/png;base64,{base64.b64encode(open('INST.LDMD.png', 'rb').read()).decode()}" 
             style="width:150%; max-height:310px; object-fit:cover; border-radius:12px;" />
    """,
    unsafe_allow_html=True
)
# ===== FULL-WIDTH HEADER BANNER =====
st.markdown(
    """
    <style>
    .full-width-banner {
        width: 100vw;
        height: 300px;
        background-image: url('Spectraa.AI/INST.LDMD.png'); 
        background-size: cover;
        background-position: center;
        margin-left: -3rem; /* removes Streamlit page padding */
        margin-top: -1rem;
    }
    </style>
    <div class="full-width-banner"></div>
    """,
    unsafe_allow_html=True
)



st.markdown("""
<style>
body {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    background-attachment: fixed;
}

div[data-testid="stRadio"] > label {
    font-size: 1rem;
    color: white;
}

h2, .stHeader, .stSubheader {
    color: #ffffff;
}

.login-container {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    width: 400px;
    margin: 0 auto;
}

.login-title {
    text-align: center;
    font-size: 1.8rem;
    font-weight: bold;
    color: #ffffff;
    margin-bottom: 1rem;
}

.login-subtitle {
    text-align: center;
    color: #ccc;
    font-size: 0.9rem;
    margin-bottom: 2rem;
}

input {
    border-radius: 10px !important;
    padding: 0.75rem !important;
    font-size: 1rem !important;
}

.stButton button {
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 1.5rem;
    font-size: 1rem;
    margin-top: 1rem;
    transition: all 0.3s ease;
}

.stButton button:hover {
    background: linear-gradient(90deg, #5563e6, #5f3ea2);
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)






st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e3c72, #2a5298, #a1c4fd);
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)






st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .header {
        text-align: center;
        font-size:40px;
        color: #000000;
        font-weight: bold;
    }
    .subheader {
        text-align: center;
        font-size:20px;
        color: #ffffff;
    }
    </style>

    <div class='header'>🔬 Spectra.AI</div>

    """,
    unsafe_allow_html=True
)


import streamlit as st
import json
import os

USERS_FILE = 'users.json'

# Function to load users from file
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

# Function to save users to file
def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

# Initialize users in session
if 'users' not in st.session_state:
    st.session_state['users'] = load_users()

def register():
    st.markdown("---")
    st.markdown("## Register New User")

    with st.form("register_form", clear_on_submit=False):
        

        new_username = st.text_input("Username", placeholder="Enter a unique username")
        new_email = st.text_input("Email Address", placeholder="Enter your email")
        new_password = st.text_input("Password", type="password", placeholder="Choose a strong password")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password")

        submit_register = st.form_submit_button("Register", use_container_width=True)

        if submit_register:
            if new_username in st.session_state['users']:
                st.error("❌ Username already exists. Please choose a different one.")
            elif new_password != confirm_password:
                st.error("❌ Passwords do not match.")
            else:
                st.session_state['users'][new_username] = {
                    'email': new_email,
                    'password': new_password
                }
                save_users(st.session_state['users'])
                st.success(f"✅ User '{new_username}' registered successfully! You can now log in.")


def login():
    st.markdown("---")
    st.markdown("## Login")

    with st.form("login_form", clear_on_submit=False):
       
        st.markdown("Please enter your credentials below:")

        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")

        submit_login = st.form_submit_button("Login", use_container_width=True)

        if submit_login:
            users = st.session_state['users']
            if username in users and users[username]['password'] == password:
                st.session_state['authenticated'] = True
                st.session_state['current_user'] = username
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")



# Authentication control
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    option = st.radio("Choose an option", ("Login", "Register"))

    if option == "Login":
        login()
    else:
        register()

    st.stop()

# ---------- Main App Interface ----------

st.success(f"Welcome, {st.session_state['current_user']}!")




# main app logic:

class RamanNet(nn.Module):
    def __init__(self, input_length=1000, window_size=50, step=25, n1=32, n2=256, embedding_dim=128, num_classes=2):
        super(RamanNet, self).__init__()
        self.window_size = window_size
        self.step = step
        self.num_windows = (input_length - window_size) // step + 1

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(window_size, n1),
                nn.BatchNorm1d(n1),
                nn.LeakyReLU()
            ) for _ in range(self.num_windows)
        ])

        self.dropout1 = nn.Dropout(0.4)
        self.summary_dense = nn.Sequential(
            nn.Linear(n1 * self.num_windows, n2),
            nn.BatchNorm1d(n2),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )

        self.embedding = nn.Sequential(
            nn.Linear(n2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        windows = []
        for i in range(self.num_windows):
            start = i * self.step
            end = start + self.window_size
            window = x[:, start:end]
            windows.append(self.blocks[i](window))
        x = torch.cat(windows, dim=1)
        x = self.dropout1(x)
        x = self.summary_dense(x)
        emb = nn.functional.normalize(self.embedding(x), p=2, dim=1)
        out = self.classifier(emb)
        return out, emb

# ---------------- Helper Functions ---------------- #
def preprocess_spectrum(df, target_len=1000):
    try:
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        x = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
        x_uniform = np.linspace(x.min(), x.max(), target_len)
        y_interp = interp1d(x, y, kind='linear', fill_value="extrapolate")(x_uniform)
        y_norm = StandardScaler().fit_transform(y_interp.reshape(-1, 1)).flatten()
        return y_norm
    except:
        return None

# ---------------- Streamlit App ---------------- #
st.set_page_config(page_title="Spectral.AI", layout="centered")
st.title("🧪 Train Spectroscopic Data")

# 🌟 Styled Upload Section
st.markdown("""
    <div style='
        background: linear-gradient(to right, #e0eafc, #cfdef3);
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        margin-bottom: 30px;
    '>
        <h3 style='text-align:center; color:#004085;'>Step 1: Upload Spectra</h3>
        <p style='text-align:center; font-size:16px; color:#333;'>Upload your pre-labeled spectra for model training.</p>
    </div>
""", unsafe_allow_html=True)


present_files = st.file_uploader("📂 Upload spectra where TARGET is PRESENT (label = 1)", type=["csv"], accept_multiple_files=True)
absent_files = st.file_uploader("📂 Upload spectra where TARGET is ABSENT (label = 0)", type=["csv"], accept_multiple_files=True)

MAX_FILES_PER_LABEL = 150

if present_files:
    if len(present_files) > MAX_FILES_PER_LABEL:
        st.error(f"❌ You can only upload up to {MAX_FILES_PER_LABEL} files. Trimming to first {MAX_FILES_PER_LABEL} files.")
        present_files = present_files[:MAX_FILES_PER_LABEL]
    st.info(f"Target Present files uploaded: {len(present_files)}/{MAX_FILES_PER_LABEL}")

if absent_files:
    if len(absent_files) > MAX_FILES_PER_LABEL:
        st.error(f"❌ You can only upload up to {MAX_FILES_PER_LABEL} files. Trimming to first {MAX_FILES_PER_LABEL} files.")
        absent_files = absent_files[:MAX_FILES_PER_LABEL]
    st.info(f"Target Absent files uploaded: {len(absent_files)}/{MAX_FILES_PER_LABEL}")





X = []
y = []

if present_files:
    st.success(f"✅ Uploaded {len(present_files)} 'Present' files")
    for file in present_files:
        df = pd.read_csv(file, header=None)
        processed = preprocess_spectrum(df)
        if processed is not None:
            X.append(processed)
            y.append(1)

          


if absent_files:
    st.success(f"✅ Uploaded {len(absent_files)} 'Absent' files")
    for file in absent_files:
        df = pd.read_csv(file, header=None)
        processed = preprocess_spectrum(df)
        if processed is not None:
            X.append(processed)
            y.append(0)

if X and y:
    st.success(f"✅ Total processed samples: {len(X)}")
    X_np = np.array(X)
    y_np = np.array(y)

  
# ⬇️ Placeholder to display training progress
progress = st.empty()

# ⬇️ This function will receive updates from inside the encrypted model_core.pyd
def update_progress(epoch, acc):
    progress.info(f"🟢 Epoch {epoch} — Train Accuracy: {acc:.2%}")

# ⬇️ When the user clicks the "Train" button
if st.button("🚀 Train Model"):
    progress.info("Training in progress...")

    # Secure call to the encrypted training function
    model = train_model(X_np, y_np, epochs=600, progress_callback=update_progress)

    # Store trained model in session for later use
    st.session_state["model"] = model

    # Final success message
    progress.success("✅ Model trained successfully")

# ---------------- Predict ---------------- #
# 🌟 Styled Step 2 Section
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
    <div style='
        background: linear-gradient(to right, #fdfbfb, #ebedee);
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-top: 30px;
        margin-bottom: 30px;
    '>
        <h3 style='text-align:center; color:#004085;'>Step 2: Test a New Spectrum</h3>
        <p style='text-align:center; font-size:16px; color:#333;'>Upload a spectrum to check if the trained model detects the target.</p>
    </div>
""", unsafe_allow_html=True)

test_file = st.file_uploader("📄 Upload test spectrum to classify", type=["csv", "txt"], key="test")

if test_file and "model" in st.session_state:
    df = pd.read_csv(test_file, header=None)

    # Optional: plot the test spectrum
    st.markdown("#### 📈 Uploaded Test Spectrum")
    try:
        df_plot = df.apply(pd.to_numeric, errors='coerce').dropna()
        fig, ax = plt.subplots()
        ax.plot(df_plot.iloc[:, 0], df_plot.iloc[:, 1])
        ax.set_xlabel("Wavenumber")
        ax.set_ylabel("Intensity")
        ax.set_title("Test Spectrum")
        st.pyplot(fig)
    except:
        st.warning("⚠️ Could not plot uploaded spectrum.")

    # ⬇️ Secure call to model_core.pyd
    pred, conf = predict_spectrum(st.session_state["model"], df)

        # Count how many files were uploaded for training
    present_count = len(present_files) if present_files else 0
    absent_count  = len(absent_files)  if absent_files else 0

    # 📥 Section for downloading the report
    st.markdown("---")
    st.subheader("📥 Download Analysis Report")

  


    if st.button("Generate PDF"):
        pdf_buffer = generate_pdf_report(present_count, absent_count, fig, pred, conf)
        st.download_button(
            label="⬇️ Download Report",
            data=pdf_buffer,
            file_name="spectra_report.pdf",
            mime="application/pdf"
        )


    # ⬇️ Styled results
    if conf < 0.65:
        st.markdown(f"""
            <div style='
                background: #fff3cd;
                color: #856404;
                padding: 20px;
                border-left: 6px solid #ffeeba;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                margin-top: 20px;
            '>
                <strong>⚠️ Prediction Uncertain:</strong> Confidence is <b>{conf:.2%}</b>. Please review the spectrum or retrain with more data.
            </div>
        """, unsafe_allow_html=True)

    elif pred == 1:
        st.markdown(f"""
            <div style='
                background: #d4edda;
                color: #155724;
                padding: 20px;
                border-left: 6px solid #28a745;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                margin-top: 20px;
            '>
                <strong>✅ Target Detected:</strong> The target is <b>present</b> in the sample with <b>{conf:.2%}</b> confidence.
            </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
            <div style='
                background: #f8d7da;
                color: #721c24;
                padding: 20px;
                border-left: 6px solid #dc3545;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                margin-top: 20px;
            '>
                <strong>❌ Target Not Detected:</strong> The target is <b>absent</b> in the sample with <b>{conf:.2%}</b> confidence.
            </div>
        """, unsafe_allow_html=True)

