import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Load env variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("❌ GOOGLE_API_KEY not found in .env file!")
    st.stop()

st.title("Task Classifier and Gemini Text Generator with Follow-Up")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=google_api_key
)

# Session-state persistent conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Task classifier prompt template (unchanged)
router_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
    You are a task classifier. 
    Based on the user query, classify it into one of the following categories:
    - "Text Task"
    - "Image Task"
    - "Other"
    Query: {query}
    Category:
    """
)
router_chain = router_prompt | llm

# Compose prompt with conversation history for context and current input
def compose_prompt(conversation, current_query):
    prompt_text = ""
    for i, (user_q, assistant_a) in enumerate(conversation):
        prompt_text += f"User: {user_q}\nAssistant: {assistant_a}\n"
    prompt_text += f"User: {current_query}\nAssistant:"
    return prompt_text

# Handler for text tasks (no web search)
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

def handle_other_task(query: str):
    return "⚠️ Sorry, I don’t know how to handle this task yet."

def route_task(conversation, query: str):
    classification = router_chain.invoke({"query": query})
    category = getattr(classification, 'content', str(classification)).strip()
    if category == "Text Task":
        result = handle_text_task(conversation, query)
    elif category == "Image Task":
        result = "Image generation is not enabled in this app."
    else:
        result = handle_other_task(query)
    return category, result

query = st.text_input("Enter your request:", key="input_query")

col1, col2 = st.columns([1, 3])
with col1:
    process_clicked = st.button("Process")
with col2:
    clear_clicked = st.button("Clear Conversation")

if clear_clicked:
    st.session_state.conversation = []
    st.experimental_rerun()

if process_clicked and query:
    cat, ans = route_task(st.session_state.conversation, query)
    # Append to conversation history
    st.session_state.conversation.append((query, ans))
    st.write(f"**Task Type:** {cat}")
    st.markdown(f"**Answer:**\n\n{ans}")

if st.session_state.conversation:
    st.markdown("### Conversation History")
    for i, (user_q, assistant_a) in enumerate(st.session_state.conversation):
        st.markdown(f"**User:** {user_q}")
        st.markdown(f"**Assistant:** {assistant_a}")
        st.markdown("---")
