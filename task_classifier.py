import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env only for local development
load_dotenv()

# Get API keys
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
hf_api_key = st.secrets.get("HF_API_KEY", os.getenv("HF_API_KEY"))

if not google_api_key:
    st.error("❌ GOOGLE_API_KEY not found! Please set it in .env (local) or Streamlit Secrets (cloud).")
    st.stop()

if not hf_api_key:
    st.warning("⚠️ Hugging Face API key not found. Image generation will not work.")

st.title("⚡ Task Classifier with Gemini + Hugging Face Images")

# ✅ Cache LLM initialization
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=google_api_key
    )

llm = load_llm()

# ✅ Session-state persistent conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# 🚀 Smarter Rule-Based Classifier
def classify_query(query: str) -> str:
    q = query.lower()
    if any(word in q for word in ["draw", "image", "picture", "diagram", "photo", "generate image"]):
        return "Image Task"
    return "Text Task"

# Compose prompt with conversation history
def compose_prompt(conversation, current_query):
    prompt_text = ""
    for user_q, assistant_a in conversation:
        prompt_text += f"User: {user_q}\nAssistant: {assistant_a}\n"
    prompt_text += f"User: {current_query}\nAssistant:"
    return prompt_text

# Handler for text tasks
def handle_text_task(conversation, query: str):
    lower_q = query.lower()
    if any(phrase in lower_q for phrase in [
        "who built this agent", "who created this agent", "who made this agent",
        "who is the developer", "who is the creator"
    ]):
        return "This agent was built by Sounak Sarkar."
    prompt = compose_prompt(conversation, query)
    response = llm.invoke(prompt)
    return getattr(response, 'content', str(response))

# ✅ Handler for Hugging Face Image tasks
def handle_image_task(prompt: str):
    if not hf_api_key:
        return "⚠️ Hugging Face API key missing. Please add it to use image generation."

    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
    headers = {"Authorization": f"Bearer {hf_api_key}"}

    try:
        response = requests.post(api_url, headers=headers, json={"inputs": prompt})
        if response.status_code != 200:
            return f"⚠️ Image generation failed: {response.text}"

        img_path = "generated.png"
        with open(img_path, "wb") as f:
            f.write(response.content)

        st.image(img_path, caption="Generated Image")
        return "✅ Image generated successfully!"
    except Exception as e:
        return f"⚠️ Error: {e}"

def handle_other_task(query: str):
    return "⚠️ Sorry, I don’t know how to handle this task yet."

def route_task(conversation, query: str):
    category = classify_query(query)
    if category == "Text Task":
        result = handle_text_task(conversation, query)
    elif category == "Image Task":
        result = handle_image_task(query)
    else:
        result = handle_other_task(query)
    return category, result

# Streamlit UI
query = st.text_input("Enter your request:", key="input_query")

col1, col2 = st.columns([1, 3])
with col1:
    process_clicked = st.button("Process")
with col2:
    clear_clicked = st.button("Clear Conversation")

if clear_clicked:
    st.session_state.conversation = []  # reset history

if process_clicked and query:
    cat, ans = route_task(st.session_state.conversation, query)
    st.session_state.conversation.append((query, ans))
    st.write(f"**Task Type:** {cat}")
    st.markdown(f"**Answer:**\n\n{ans}")

if st.session_state.conversation:
    st.markdown("### Conversation History")
    for user_q, assistant_a in st.session_state.conversation:
        st.markdown(f"**User:** {user_q}")
        st.markdown(f"**Assistant:** {assistant_a}")
        st.markdown("---")
