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
    <p><strong>Spectral.AI</strong> is an advanced machine learning application purpose-built to train on raw spectral data and accurately differentiate even the most complex and closely resembling spectra. By leveraging state-of-the-art algorithms, it can identify analytes from minute trace-level spectral signatures with exceptional accuracy.

Extensively validated in laboratory settings, Spectral.AI has demonstrated outstanding performance in analyzing Raman signals, reliably distinguishing target molecular signatures within highly complex systems. It has been successfully optimized for biomarkers such as PCT, IL-3, and CRP, achieving detection limits down to ~100 fM concentrations.

Highly adaptable, the platform can be trained for a wide range of analytes and applied across multiple spectroscopic techniques. As the first application of its kind to integrate machine learning directly with raw spectroscopic data, Spectral.AI sets a new benchmark for precision analytics and scientific innovation.

If you want, I can also elevate this further into a high-impact premium version tailored for your Spectra.AI homepage or research report so it sounds both cutting-edge and visionary.</p>

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
