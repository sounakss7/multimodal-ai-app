import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load local .env (for development only)
load_dotenv()

# Fetch API key (Streamlit Cloud > Secrets OR local .env)
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

if not google_api_key:
    st.error("❌ GOOGLE_API_KEY not found! Please set it in .env (local) or Streamlit Secrets (cloud).")
    st.stop()

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=google_api_key)

# Streamlit UI
st.title("🔮 Task Classifier Agent")

prompt = st.text_area("Enter your task prompt:")

if st.button("Classify"):
    if prompt.strip():
        with st.spinner("Classifying..."):
            try:
                response = llm.invoke(prompt)
                st.success("✅ Result:")
                st.write(response.content)
            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")
    else:
        st.warning("Please enter a prompt before classifying.")
