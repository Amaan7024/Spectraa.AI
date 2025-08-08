import streamlit as st

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #fff9db !important;
        color: black !important;
    }

    [data-testid="stSidebar"] * {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- Global Styling ---
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
        color: #f0f0f0;
    }
    .about-container {
        background-color: rgba(255, 255, 255, 0.07);
        padding: 2.5rem;
        border-radius: 20px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        max-width: 950px;
        margin: 2rem auto;
        animation: fadeIn 1.2s ease-in-out;
    }
    h1 {
        font-size: 2.5rem;
        text-align: center;
        color: white;
        text-shadow: 0 0 15px #00c3ff;
    }
    h2, h3 {
        color: #ffffff;
        margin-top: 2rem;
    }
    p {
        font-size: 16px;
        color: #dddddd;
        line-height: 1.6;
        text-align: justify;
    }
    ul, ol {
        font-size: 16px;
        color: #dddddd;
        line-height: 1.6;
    }
    .highlight-box {
        background: rgba(255,255,255,0.08);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    .highlight-box:hover {
        transform: scale(1.02);
    }
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(to right, #ffffff20, #ffffff60, #ffffff20);
        margin: 2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- All content inside the container ----------
st.markdown("<div class='about-container'>", unsafe_allow_html=True)

st.markdown("<h1>📖 About Spectra.AI</h1>", unsafe_allow_html=True)

st.markdown("""
Spectra.AI is an intelligent, deep learning–powered web application designed for rapid analysis and classification of **a spectra**.  
It empowers researchers, students, and analysts to detect the presence or absence of a target from spectral data — without needing coding or ML expertise.
""")

# Why Spectra.AI
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 🎯 Why Spectra.AI?")
st.markdown("""
<div class='highlight-box'>
<ul>
<li>With Spectra.AI, just upload your labeled data and instantly train an advanced model.</li>
<li>The app supports <b>binary classification</b> with real-time accuracy feedback, uncertainty alerts, and result confidence.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Key Features
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 🌟 Key Features")
st.markdown("""
<div class='highlight-box'>
<ul>
<li>🔐 Secure <b>login & registration</b> system with local storage.</li>
<li>⚙️ Custom-built deep learning model optimized for 1D signals.</li>
<li>🧠 Live training with confidence updates at each epoch.</li>
<li>📈 Upload & classify new test samples with interpretability.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# How It Works
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 🧪 How It Works")
st.markdown("""
<div class='highlight-box'>
<ol>
<li>Upload labeled <b>spectra files</b> (Target Present & Target Absent).</li>
<li>Click <b>Train Model</b> — RamanNet trains on uploaded data with live accuracy feedback.</li>
<li>Upload a new spectrum for prediction.</li>
<li>Get a result with <b>output</b>, <b>confidence level</b>, and <b>uncertainty warning</b> if needed.</li>
</ol>
</div>
""", unsafe_allow_html=True)

# Acknowledgements
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 🙏 Acknowledgements")
st.markdown("""
<div class='highlight-box'>
This project acknowledges the foundational contributions of:
<ul>
<li><b>RamanNet</b>: A lightweight convolutional neural network for Raman spectral classification (2022)</li>
<li><b>Modified RamanNet</b>: An improved version with optimized architecture for enhanced detection (2025)</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

