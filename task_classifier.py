import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import requests
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Get API keys
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
hf_api_key = st.secrets.get("HF_API_KEY", os.getenv("HF_API_KEY"))

# Initialize Gemini model
if google_api_key:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
else:
    llm = None

# Prompt template to classify tasks
task_classifier_prompt = PromptTemplate(
    input_variables=["task"],
    template="Classify the following task as either a 'Text Task' or 'Image Task': {task}"
)

# Function: handle text tasks with Gemini
def handle_text_task(task):
    if not llm:
        return "⚠️ Google Gemini API key missing."
    response = llm.predict(task)
    return response

# Function: handle image tasks with Hugging Face
def handle_image_task(prompt: str):
    if not hf_api_key:
        return "⚠️ Hugging Face API key missing. Please add it to use image generation."
    api_url = "https://api-inference.huggingface.co/models/prompthero/openjourney"
    headers = {"Authorization": f"Bearer {hf_api_key}"}
    try:
        response = requests.post(api_url, headers=headers, json={"inputs": prompt})
        content_type = response.headers.get("content-type")
        if content_type == "application/json":
            error_msg = response.json()
            return f"⚠️ Hugging Face error: {error_msg.get('error', error_msg)}"
        # Try decoding image bytes
        try:
            image = Image.open(io.BytesIO(response.content))
            st.image(image, caption=f"Generated: {prompt}")
            return "✅ Image generated successfully!"
        except Exception as img_err:
            return f"⚠️ Error decoding image: {img_err}.\nRaw response headers: {response.headers}\nRaw response bytes: {response.content[:100]}"
    except Exception as e:
        return f"⚠️ Error: {e}"

# Streamlit UI
st.set_page_config(page_title="Task Classifier", page_icon="⚡", layout="centered")
st.title("⚡ Task Classifier with Gemini + Hugging Face Images")

user_input = st.text_input("Enter your request:")

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Process"):
    if user_input:
        # Classify the task
        classification_prompt = task_classifier_prompt.format(task=user_input)
        task_type = llm.predict(classification_prompt) if llm else "Text Task"
        # Decide based on task type
        if "image" in task_type.lower():
            result = handle_image_task(user_input)
        else:
            result = handle_text_task(user_input)
        st.session_state.history.append((user_input, result))
        st.subheader("Task Type:")
        st.write(task_type)
        st.subheader("Answer:")
        st.write(result)

if st.button("Clear Conversation"):
    st.session_state.history = []

st.subheader("Conversation History ↔")
for i, (q, a) in enumerate(st.session_state.history, 1):
    st.write(f"**User:** {q}")
    st.write(f"**Assistant:** {a}")
