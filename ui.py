import streamlit as st
import numpy as np
from PIL import Image

from app import detect_objects

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="🐄 Animal AI System", layout="wide")

# -------------------------------
# HEADER
# -------------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#4CAF50;'>
    🐄 Animal Detection & Measurement AI
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("---")

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload Animal Image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file)
    img_array = np.array(image)

    col1, col2 = st.columns(2)

    # -------------------------------
    # ORIGINAL IMAGE
    # -------------------------------
    with col1:
        st.markdown("### 🖼 Original Image")
        st.image(image, use_container_width=True)

    # -------------------------------
    # DETECTION + MEASUREMENTS
    # -------------------------------
    result_img, data = detect_objects(img_array.copy())

    with col2:
        st.markdown("### 🔍 Detected Image")
        st.image(result_img, use_container_width=True)

    st.write("---")

    # -------------------------------
    # ANALYSIS SECTION
    # -------------------------------
    st.markdown("### 📊 Animal Measurements")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📏 Body Length", f"{data['body_length']} px")
        st.metric("📏 Height", f"{data['height']} px")

    with col2:
        st.metric("🦵 Leg Length", f"{data['leg_length']} px")
        st.metric("🫁 Chest Width", f"{data['chest_width']} px")

    with col3:
        st.metric("📐 Rump Angle", f"{data['rump_angle']}°")
        st.metric("🐄 Breed", data['breed'])

    st.write("---")

    # -------------------------------
    # INFO BOX
    # -------------------------------
    st.info(
        "⚠️ These values are estimated using image geometry (bounding box). "
        "For real-world accuracy, use pose estimation or calibration."
    )

# -------------------------------
# FOOTER
# -------------------------------
st.write("---")
st.markdown(
    "<p style='text-align:center;color:gray;'>🚀 Powered by YOLOv8 + Streamlit</p>",
    unsafe_allow_html=True
)