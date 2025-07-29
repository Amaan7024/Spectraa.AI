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


from PIL import Image
import os

st.set_page_config(page_title="Spectra.AI - Team", layout="centered")

# Optional: Match background to main app
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
        color: white;
    }
    .member-section {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    .member-image {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    a {
        color: #a1c4fd;
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("👥 Meet the Team")
st.markdown("We’re a group of passionate researchers, developers, and innovators behind Spectra.AI.")

# ----------- Helper function for each member -----------
def display_member(image_path, name, role, bio, email=None, linkedin=None):
    with st.container():
        col1, col2 = st.columns([1, 3])
        with col1:
            if os.path.exists(image_path):
                st.image(image_path, caption=name, use_container_width=True)
            else:
                st.warning(f"Image not found: `{image_path}`")
        with col2:
            st.markdown(f"### {name}")
            st.markdown(f"**{role}**")
            st.markdown(bio)

            if email:
                st.markdown(f"📧 Email: [{email}](mailto:{email})", unsafe_allow_html=True)
            if linkedin:
                st.markdown(f"🔗 LinkedIn: [{linkedin}]({linkedin})", unsafe_allow_html=True)

        st.markdown("---")

# ----------- Add Members Here -----------
display_member(
    "team_photos/1.png",
    "Dr. Kiran Shankar Hazra",
    "Supervisor",
    "Scientist E, INST Mohali",
    email="kiran@inst.ac.in",
    
)

display_member(
    "team_photos/2.png",
    "Mohd. Amaan Ansari",
    "Machine Learning Lead",
    "Intern, INST Mohali",
    
)

display_member(
    "team_photos/3.png",
    "Manish Singh Pawar",
    "Lead Data Researcher",
    "Project Associate, INST Mohali",
    
)

display_member(
    "team_photos/4.png",
    "Mohd. Zaid Habib",
    "Lead Analyst",
    "Student, Panjab University",
   
)




