import streamlit as st

# Sidebar style
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

# Global UI styling
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
        font-size: 2.7rem;
        text-align: center;
        color: white;
        text-shadow: 0 0 20px #00c3ff;
    }
    h2, h3 {
        color: #ffffff;
        margin-top: 2rem;
    }
    p, li {
        font-size: 16px;
        color: #dddddd;
        line-height: 1.6;
        text-align: justify;
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
    .banner {
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    .banner img {
        width: 100px;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 0 8px #00c3ff);
    }
    .banner h1 {
        margin-top: 0;
        font-size: 2.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Content ----------
st.markdown("<div class='about-container'>", unsafe_allow_html=True)

# Banner Section
st.markdown("""
<div class='banner'>
    <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/React-icon.svg/512px-React-icon.svg.png' alt='Spectra.AI Logo' />
    <h1>📖 About Spectra.AI</h1>
</div>
""", unsafe_allow_html=True)

# About description
st.markdown("""
Spectra.AI is an intelligent, deep learning–powered web application designed for rapid analysis and classification of **spectral data**.  
It empowers researchers, students, and analysts to detect the presence or absence of a target from spectral inputs — without needing coding or ML expertise.
""")

# Why Spectra.AI
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 🎯 Why Spectra.AI?")
st.markdown("""
<div class='highlight-box'>
<ul>
<li>Upload your labeled data and instantly train an advanced model.</li>
<li>Supports <b>binary classification</b> with real-time feedback and uncertainty detection.</li>
<li>Analyze any spectra easily within seconds.</li>
<li>No need to write code — fully automated UI-driven system.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Key Features
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 🌟 Key Features")
st.markdown("""
<div class='highlight-box'>
<ul>
<li>🔐 Secure login & registration system.</li>
<li>⚙️ Custom deep learning model optimized for 1D signals.</li>
<li>📡 Live model training with accuracy and confidence updates.</li>
<li>📂 Upload & classify new spectra with uncertainty warnings.</li>
<li>🧊 Glassmorphism UI for clean, modern experience.</li>
<li>📊 Download model summary and predictions.</li>
<li>📁 Accepts CSV, TXT files with automatic parsing.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# How it works
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 🧪 How It Works")
st.markdown("""
<div class='highlight-box'>
<ol>
<li>Firstly Upload <b>spectra files</b> for both Target Present and Target Absent classes, i.e., upload some spectra in which the target is present and some in which spectra is absent.</li>
<li>The data can be in .csv or.txt format. In each section you can upload a maximum of 150 files. More than 100 files in each section are preferred for better results.</li>
<li>Click <b>Train Model</b> — The model gets traind  and reports accuracy each epoch.It will run for 600 epochs.After training, a message will be displayed indicating that the model is trained.</li>
<li>Then, in step 2 ,Upload a new test spectrum for prediction. You will see a visual of the test spectrum you are checking.</li>
<li>Get results with <b>confidence level</b> and <b>uncertainty alerts</b>.</li>
</ol>
</div>
""", unsafe_allow_html=True)

# New Additions
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 🚀 What's New")
st.markdown("""
<div class='highlight-box'>
<ul>
<li>🧠 <b>Uncertainty-based decision support</b> — warns user if model is unsure.</li>
<li>📈 <b>Live training plots</b> of accuracy vs. epochs.</li>
<li>💾 <b>Model encrypted</b> with secure .so/.pyd deployment — protecting IP.</li>
<li>🧩 Modular pages (Home, About, Team, Train) for seamless UX.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Acknowledgements
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 🙏 Acknowledgements")
st.markdown("""
<div class='highlight-box'>
This project acknowledges:
<ul>
<li><b>RamanNet (2022)</b>: A lightweight CNN for Raman spectral classification</li>
<li><b>Modified RamanNet (2025)</b>: Enhanced version used in Spectra.AI</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

