import os
import requests
import base64
from io import BytesIO
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai   # ✅ Use Google SDK instead of LangChain
from langchain.schema import HumanMessage

# =====================
# Load environment variables
# =====================
load_dotenv()
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
pollinations_token = st.secrets.get("POLLINATIONS_TOKEN", os.getenv("POLLINATIONS_TOKEN"))

if not google_api_key:
    st.error("❌ GOOGLE_API_KEY not found! Please set it in .env or Streamlit Secrets.")
    st.stop()

if not pollinations_token:
    st.error("❌ POLLINATIONS_TOKEN not found! Please set it in .env or Streamlit Secrets.")
    st.stop()

# =====================
# Initialize Gemini Flash Model
# =====================
genai.configure(api_key=google_api_key)
flash_model = genai.GenerativeModel("gemini-1.5-flash")  # ✅ official supported model

def flash_stream(prompt):
    """Streaming response from Gemini Flash"""
    response = flash_model.generate_content(prompt, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text

def flash_invoke(prompt):
    """Single response (non-streaming) from Gemini Flash"""
    response = flash_model.generate_content(prompt)
    return response.text

# =====================
# Streamlit App Layout
# =====================
st.set_page_config(page_title="🤖 Multimodal AI App", page_icon="🤖", layout="centered")
st.title("🤖 Multimodal AI App (Text + Image Generator + Image Q&A)")

tab1, tab2, tab3 = st.tabs(["💬 Text Chat", "🎨 Image Generator", "🖼️ Image Q&A"])

# =====================
# TEXT CHAT TAB
# =====================
with tab1:
    st.subheader("⚡ Fast Text Task Classifier & Gemini Chat")

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    def compose_prompt(conversation, current_query):
        prompt_text = ""
        for user_q, assistant_a in conversation:
            prompt_text += f"User: {user_q}\nAssistant: {assistant_a}\n"
        prompt_text += f"User: {current_query}\nAssistant:"
        return prompt_text

    def handle_text_task(conversation, query: str):
        lower_q = query.lower()
        if any(phrase in lower_q for phrase in [
            "who built this agent", "who created this agent", "who made this agent",
            "who is the developer", "who is the creator"
        ]):
            return "This agent was built by **Sounak Sarkar**."

        prompt = compose_prompt(conversation, query)

        response_placeholder = st.empty()
        final_response = ""

        for chunk in flash_stream(prompt):  # ✅ use flash_stream
            final_response += chunk
            response_placeholder.markdown(f"**Answer (streaming):**\n\n{final_response}")

        return final_response

    query = st.text_input("💬 Enter your request:", key="input_query")

    col1, col2 = st.columns([1, 3])
    with col1:
        process_clicked = st.button("Process")
    with col2:
        clear_clicked = st.button("Clear Conversation")

    if clear_clicked:
        st.session_state.conversation = []

    if process_clicked and query:
        ans = handle_text_task(st.session_state.conversation, query)
        st.session_state.conversation.append((query, ans))

    if st.session_state.conversation:
        st.markdown("### 🗂️ Conversation History")
        for user_q, assistant_a in st.session_state.conversation:
            st.markdown(f"**User:** {user_q}")
            st.code(assistant_a, language="markdown")
            st.markdown("---")

# =====================
# IMAGE GENERATOR TAB
# =====================
with tab2:
    st.subheader("🎨 Pollinations.AI Free Image Generator")

    img_prompt = st.text_input("📝 Enter your image prompt:", key="img_prompt")

    # Style options
    styles = ["Realistic", "Cartoon", "Fantasy", "Minimalist"]
    selected_style = st.radio("🎨 Choose a style:", styles, horizontal=True)

    # Function: Ask Gemini Flash to expand + improve the prompt
    def smart_enhance_prompt(user_prompt, style):
        quick_prompt = f"Rewrite this short prompt into a detailed {style} image generation description: {user_prompt}"
        return flash_invoke(quick_prompt).strip()

    @st.cache_data(show_spinner=False)
    def fetch_image(final_prompt, token):
        url = f"https://image.pollinations.ai/prompt/{final_prompt}?token={token}"
        return requests.get(url).content

    if st.button("Generate Image"):
        if not img_prompt:
            st.warning("⚠️ Please enter a prompt before generating an image.")
        else:
            with st.spinner(f"🎨 Generating {selected_style} image..."):
                try:
                    final_prompt = smart_enhance_prompt(img_prompt, selected_style)
                    img_bytes = fetch_image(final_prompt, pollinations_token)
                    img = Image.open(BytesIO(img_bytes))
                    st.image(img, caption=final_prompt)

                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    st.download_button(
                        label="📥 Download Image",
                        data=buf.getvalue(),
                        file_name="pollinations_image.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"❌ Failed to generate image: {e}")

# =====================
# IMAGE Q&A TAB
# =====================
with tab3:
    st.subheader("🖼️ Upload an Image & Ask Gemini")

    uploaded_img = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"])
    qna_prompt = st.text_input("💬 Ask something about the uploaded image:")

    if st.button("Analyze Image"):
        if not uploaded_img:
            st.warning("⚠️ Please upload an image first.")
        elif not qna_prompt:
            st.warning("⚠️ Please enter a question about the image.")
        else:
            with st.spinner("🔎 Analyzing image..."):
                img_bytes = uploaded_img.read()
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                data_url = f"data:image/png;base64,{img_base64}"

                # Pass both image and question to Gemini Flash
                content = [
                    {"type": "text", "text": qna_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]

                response_placeholder = st.empty()
                final_response = ""

                response = flash_model.generate_content(content, stream=True)  # ✅ Flash supports multimodal
                for chunk in response:
                    if chunk.text:
                        final_response += chunk.text
                        response_placeholder.markdown(f"**Answer (streaming):**\n\n{final_response}")
