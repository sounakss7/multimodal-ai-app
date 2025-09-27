import os
import requests
import base64
from io import BytesIO
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
import urllib.parse # Required for the URL encoding fix

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
# Initialize Gemini LLM (⚡ streaming mode)
# =====================
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",  # FIXED: Use a widely available and powerful model
    temperature=0,
    google_api_key=google_api_key,
    streaming=True
)

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

        # FIXED: Wrap the prompt in HumanMessage for consistent API calls
        for chunk in llm.stream([HumanMessage(content=prompt)]):
            final_response += chunk.content or ""
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

    styles = ["Realistic", "Cartoon", "Fantasy", "Minimalist", "Anime"]
    selected_style = st.radio("🎨 Choose a style:", styles, horizontal=True)

    def smart_enhance_prompt(user_prompt, style):
        quick_prompt = f"Rewrite this short prompt into a detailed, single-paragraph {style} style image generation description suitable for an AI image generator: {user_prompt}"
        response = llm.invoke(quick_prompt)
        return response.content.strip()

    @st.cache_data(show_spinner=False)
    def fetch_image(final_prompt, token):
        # FIXED: URL-encode the prompt to handle spaces and special characters
        encoded_prompt = urllib.parse.quote(final_prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?token={token}"
        return requests.get(url).content

    if st.button("Generate Image"):
        if not img_prompt:
            st.warning("⚠️ Please enter a prompt before generating an image.")
        else:
            with st.spinner(f"🎨 Generating {selected_style} image..."):
                final_prompt = smart_enhance_prompt(img_prompt, selected_style)
                st.info(f"Enhanced Prompt: {final_prompt}")

                try:
                    img_bytes = fetch_image(final_prompt, pollinations_token)
                    img = Image.open(BytesIO(img_bytes))
                    st.image(img, caption=final_prompt)

                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    st.download_button(
                        label="📥 Download Image",
                        data=buf.getvalue(),
                        file_name="generated_image.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"❌ Failed to generate image. The API may be busy. Please try again. Error: {e}")


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
                
                # FIXED: Use the actual MIME type from the uploaded file
                mime_type = uploaded_img.type
                data_url = f"data:{mime_type};base64,{img_base64}"

                content = [
                    {"type": "text", "text": qna_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]

                response_placeholder = st.empty()
                final_response = ""

                for chunk in llm.stream([HumanMessage(content=content)]):
                    final_response += chunk.content or ""
                    response_placeholder.markdown(f"**Answer (streaming):**\n\n{final_response}")
