import streamlit as st

st.set_page_config(page_title="About - Spectra.AI", layout="centered")

st.markdown(
    """
    <style>
    .about-title {
        font-size: 40px;
        font-weight: 800;
        text-align: center;
        color: #3a47d5;
        margin-top: 20px;
    }
    .about-content {
        font-size: 18px;
        text-align: justify;
        margin: 30px auto;
        max-width: 800px;
        line-height: 1.8;
        color: #333;
        background: #ffffffcc;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 0 12px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='about-title'>🔍 About Spectra.AI</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class='about-content'>
    <p><strong>Spectra.AI</strong> is an intelligent Raman spectra analysis platform. It enables users to train deep learning models directly on their uploaded spectral data — no coding required.</p>

    <p>Here's how it works:</p>
    <ol>
        <li><strong>Upload Training Data:</strong> The user provides two sets of spectral files: one where the target is <b>present</b> and another where it's <b>absent</b>.</li>
        <li><strong>Train Your Model:</strong> With a single click, Spectra.AI trains a custom RamanNet model on this data.</li>
        <li><strong>Predict on New Data:</strong> Upload a new spectrum and the app will instantly tell whether the target is likely present or not — with confidence.</li>
    </ol>

    <p>💡 Designed for researchers, scientists, and diagnostic professionals who want quick, powerful insights from their spectral data.</p>
    </div>
    """,
    unsafe_allow_html=True
)
