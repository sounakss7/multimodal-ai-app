import os
import streamlit as st
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env only for local development
load_dotenv()

# Get API keys (Streamlit Secrets first, fallback to .env)
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
fal_api_key = st.secrets.get("FAL_API_KEY", os.getenv("FAL_API_KEY"))

if not google_api_key:
    st.error("❌ GOOGLE_API_KEY not found! Please set it in .env or Streamlit Secrets.")
    st.stop()

if not fal_api_key:
    st.warning("⚠️ FAL_API_KEY not found! Image tasks will not work.")

st.title("⚡ Task Classifier with Gemini + FAL AI Images")

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

# 🚀 Rule-Based Classifier
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

# ✅ Handler for FAL image generation
def handle_image_task(query: str):
    if not fal_api_key:
        return "❌ FAL API key not set. Please configure it to enable image generation."

    api_url = "https://api.fal.ai/v1/run/stable-diffusion-xl"
    headers = {
        "Authorization": f"Key {fal_api_key}",
        "Content-Type": "application/json"
    }
    payload = {"prompt": query}

    with st.spinner("🎨 Generating image with FAL AI..."):
        response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        image_url = result.get("images", [{}])[0].get("url", None)

        if image_url:
            st.image(image_url, caption=f"Generated for: {query}")
            return "✅ Image generated successfully!"
        else:
            return "⚠️ No image returned. Try another prompt."
    else:
        return f"⚠️ Image generation failed: {response.text}"

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
    st.rerun()

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
