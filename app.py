import streamlit as st
import sys
import os

# Add model folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))

from predict import predict

st.set_page_config(
    page_title="Cyberbullying Detection System",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Cyberbullying Detection System")
st.write("### PyTorch + NLP + LSTM Neural Network")

user_input = st.text_area(
    "Enter a comment:",
    height=150,
    placeholder="Type your comment here..."
)

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:

        result, confidence = predict(user_input)

        if result == "Cyberbullying":
            st.error(f"🚨 {result}")
        else:
            st.success(f"✅ {result}")

        st.write(f"**Confidence:** {confidence:.4f}")