import os
import requests
import base64
from io import BytesIO
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from audio_recorder_streamlit import audio_recorder # 🎙️ for mic input

# =====================
# Load environment variables
# =====================
load_dotenv()
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
pollinations_token = st.secrets.get("POLLINATIONS_TOKEN", os.getenv("POLLINATIONS_TOKEN"))
gladia_api_key = st.secrets.get("GLADIA_API_KEY", os.getenv("GLADIA_API_KEY"))

if not google_api_key:
    st.error("❌ GOOGLE_API_KEY not found! Please set it in .env or Streamlit Secrets.")
    st.stop()

if not pollinations_token:
    st.error("❌ POLLINATIONS_TOKEN not found! Please set it in .env or Streamlit Secrets.")
    st.stop()

if not gladia_api_key:
    st.error("❌ GLADIA_API_KEY not found! Please set it in .env or Streamlit Secrets.")
    st.stop()

# =====================
# Initialize Gemini LLM
# =====================
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", # Updated to a common, effective model
    temperature=0,
    google_api_key=google_api_key,
    streaming=True
)

# =====================
# Streamlit App Layout
# =====================
st.set_page_config(page_title="🤖 Multimodal AI App", page_icon="🤖", layout="centered")
st.title("🤖 Multimodal AI App (Text + Image Generator + Image Q&A + Voice Input)")

tab1, tab2, tab3 = st.tabs(["💬 Text & Voice Chat", "🎨 Image Generator", "🖼️ Image Q&A"])

# =====================
# TEXT + VOICE CHAT TAB
# =====================
with tab1:
    st.subheader("⚡ Fast Text + Voice Task Classifier & Gemini Chat")

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

        for chunk in llm.stream(prompt):
            final_response += chunk.content or ""
            response_placeholder.markdown(f"**Answer (streaming):**\n\n{final_response}")

        return final_response

    # =====================
    # 🎙️ Voice Input + Text Input Section
    # =====================
    query = st.text_input("💬 Enter your request:", key="input_query")

    # Record voice (audio_recorder creates mic button)
    st.write("🎙️ Speak your query below:")
    audio_bytes = audio_recorder(text="", recording_color="#FF4B4B", neutral_color="#4B9EFF")
    
    # Corrected Voice Input Block
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

        with st.spinner("🎧 Transcribing your voice via Gladia..."):
            files = {'audio': ("voice.wav", audio_bytes, "audio/wav")}
            headers = {"x-gladia-key": gladia_api_key}
            response = requests.post(
                "https://api.gladia.io/audio/text/audio-transcription/",
                headers=headers,
                files=files
            )

            if response.status_code == 200:
                result_json = response.json()
                
                # --- START OF FIX ---
                # More robust logic to extract transcription text
                text_result = ""
                if isinstance(result_json, dict):
                    # Handles formats like {'prediction': 'text'} or {'transcription': 'text'}
                    if "prediction" in result_json:
                        text_result = result_json["prediction"]
                    elif "transcription" in result_json:
                        text_result = result_json["transcription"]
                elif isinstance(result_json, list) and len(result_json) > 0:
                    # Handles the format seen in your error: [{'transcription': 'text', ...}]
                    first_item = result_json[0]
                    if isinstance(first_item, dict) and "transcription" in first_item:
                        text_result = first_item.get("transcription", "")
                # --- END OF FIX ---

                if text_result:
                    st.success(f"🗣️ You said: {text_result}")
                    query = text_result.strip() # This will now work correctly on the extracted string

                    # 🔥 Directly send to Gemini
                    st.info("🤖 Sending transcribed text to Gemini...")
                    ans = handle_text_task(st.session_state.conversation, query)
                    st.session_state.conversation.append((query, ans))
                else:
                    st.warning("⚠️ Could not extract transcription from API response. Try again.")
                    st.json(result_json) # Show the raw response for debugging
            else:
                st.error(f"❌ Gladia API Error (Status {response.status_code}): {response.text}")


    col1, col2 = st.columns([1, 3])
    with col1:
        process_clicked = st.button("Process")
    with col2:
        clear_clicked = st.button("Clear Conversation")

    if clear_clicked:
        st.session_state.conversation = []
        st.rerun() # Use rerun to clear the UI instantly

    if process_clicked and query:
        ans = handle_text_task(st.session_state.conversation, query)
        st.session_state.conversation.append((query, ans))

    if st.session_state.conversation:
        st.markdown("### 🗂️ Conversation History")
        for user_q, assistant_a in st.session_state.conversation:
            st.markdown(f"**User:** {user_q}")
            st.markdown(f"**Assistant:**\n{assistant_a}")
            st.markdown("---")

# =====================
# IMAGE GENERATOR TAB
# =====================
with tab2:
    st.subheader("🎨 Pollinations.AI Free Image Generator")

    img_prompt = st.text_input("📝 Enter your image prompt:", key="img_prompt")

    styles = ["Realistic", "Cartoon", "Fantasy", "Minimalist", "Cyberpunk"]
    selected_style = st.radio("🎨 Choose a style:", styles, horizontal=True)

    def smart_enhance_prompt(user_prompt, style):
        quick_prompt = f"Rewrite this short prompt into a detailed, comma-separated, high-quality {style} image generation description for an AI: {user_prompt}"
        response = llm.invoke(quick_prompt)
        return response.content.strip()

    @st.cache_data(show_spinner=False)
    def fetch_image(final_prompt):
        # URL encoding the prompt is safer for API calls
        from urllib.parse import quote
        encoded_prompt = quote(final_prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        return requests.get(url).content

    if st.button("Generate Image"):
        if not img_prompt:
            st.warning("⚠️ Please enter a prompt before generating an image.")
        else:
            with st.spinner(f"🎨 Generating {selected_style} image..."):
                final_prompt = smart_enhance_prompt(img_prompt, selected_style)
                st.info(f"Enhanced Prompt: {final_prompt}")
                try:
                    img_bytes = fetch_image(final_prompt)
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

    if uploaded_img:
        st.image(uploaded_img, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze Image"):
        if not uploaded_img:
            st.warning("⚠️ Please upload an image first.")
        elif not qna_prompt:
            st.warning("⚠️ Please enter a question about the image.")
        else:
            with st.spinner("🔎 Analyzing image..."):
                # Reset the file pointer before reading
                uploaded_img.seek(0)
                img_bytes = uploaded_img.read()
                
                # Directly use bytes with the vision model
                image_part = {
                    "mime_type": uploaded_img.type,
                    "data": img_bytes
                }
                
                prompt_parts = [qna_prompt, image_part]

                response_placeholder = st.empty()
                final_response = ""

                # For multimodal, you pass the parts directly to the model
                model_pro = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=google_api_key)
                response = model_pro.invoke(prompt_parts)
                response_placeholder.markdown(f"**Answer:**\n\n{response.content}")
