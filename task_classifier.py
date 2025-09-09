import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Get Google Gemini API key
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

if not google_api_key:
    st.error("❌ GOOGLE_API_KEY not found! Please set it in .env (local) or Streamlit Secrets (cloud).")
    st.stop()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=google_api_key
)

# Streamlit App
st.set_page_config(page_title="⚡ Text Task Classifier", page_icon="📝", layout="centered")
st.title("⚡ Text Task Classifier & Gemini Chat")

# Conversation state
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Compose prompt with conversation history
def compose_prompt(conversation, current_query):
    prompt_text = ""
    for user_q, assistant_a in conversation:
        prompt_text += f"User: {user_q}\nAssistant: {assistant_a}\n"
    prompt_text += f"User: {current_query}\nAssistant:"
    return prompt_text

# Handle text queries
def handle_text_task(conversation, query: str):
    lower_q = query.lower()
    if any(phrase in lower_q for phrase in [
        "who built this agent", "who created this agent", "who made this agent",
        "who is the developer", "who is the creator"
    ]):
        return "This agent was built by **Sounak Sarkar**."
    prompt = compose_prompt(conversation, query)
    response = llm.invoke(prompt)
    return getattr(response, 'content', str(response))

# Streamlit UI
query = st.text_input("💬 Enter your request:", key="input_query")

col1, col2 = st.columns([1, 3])
with col1:
    process_clicked = st.button("Process")
with col2:
    clear_clicked = st.button("Clear Conversation")

if clear_clicked:
    st.session_state.conversation = []  # reset history

if process_clicked and query:
    ans = handle_text_task(st.session_state.conversation, query)
    st.session_state.conversation.append((query, ans))
    st.markdown(f"**Answer:**\n\n{ans}")

# Show history
if st.session_state.conversation:
    st.markdown("### 🗂️ Conversation History")
    for user_q, assistant_a in st.session_state.conversation:
        st.markdown(f"**User:** {user_q}")
        st.markdown(f"**Assistant:** {assistant_a}")
        st.markdown("---")
