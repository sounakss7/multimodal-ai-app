import os
import requests
import base64
from io import BytesIO
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from audio_recorder_streamlit import audio_recorder  # 🎙️ for mic input
import wave   
import io
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
import concurrent.futures


# =====================
# Load environment variables
# =====================
load_dotenv()
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
pollinations_token = st.secrets.get("POLLINATIONS_TOKEN", os.getenv("POLLINATIONS_TOKEN"))
gladia_api_key = st.secrets.get("GLADIA_API_KEY", os.getenv("GLADIA_API_KEY"))
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))


if not google_api_key:
    st.error("❌ GOOGLE_API_KEY not found! Please set it in .env or Streamlit Secrets.")
    st.stop()

if not pollinations_token:
    st.error("❌ POLLINATIONS_TOKEN not found! Please set it in .env or Streamlit Secrets.")
    st.stop()

if not gladia_api_key:
    st.error("❌ GLADIA_API_KEY not found! Please set it in .env or Streamlit Secrets.")
    st.stop()
if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found! Please set it.env or Streamlit Secrets")
    st.stop()

# =====================
# Initialize Gemini LLM
# =====================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=google_api_key,
    streaming=True
)
def choose_groq_model(prompt: str):
    p = prompt.lower()
    if any(x in p for x in ["python", "code", "algorithm", "bug", "function", "script"]):
        return "llama3-70b-8192"
    elif any(x in p for x in ["story", "poem", "creative", "write", "blog", "lyrics"]):
        return "mixtral-8x7b-32768"
    else:
        return "gemma2-9b-it"
# =====================
# Groq Text Generation (API call)
# =====================
def query_groq(prompt: str):
    model = choose_groq_model(prompt)
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": 1024,
    }
    try:
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            return f"**Model:** {model}\n\n{content}"
        else:
            return f"❌ Groq API Error: {resp.text}"
    except Exception as e:
        return f"⚠️ Groq Error: {e}"




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
    st.subheader("⚡ Gemini + Groq Parallel Response Chat")

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    query = st.text_input("💬 Enter your question:", key="input_query")

    # 🎙️ Voice input
    st.write("🎙️ Speak your query below (up to 30 seconds):")
    audio_bytes = audio_recorder(text="Click to start/stop recording", recording_color="#FF4B4B")

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        tmp_path = "temp_audio.wav"
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)

        with wave.open(tmp_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)

        if duration >= 2:
            with st.spinner("🎧 Transcribing voice..."):
                files = {'audio': ("voice.wav", audio_bytes, "audio/wav")}
                headers = {"x-gladia-key": gladia_api_key}
                response = requests.post("https://api.gladia.io/audio/text/audio-transcription/", headers=headers, files=files)
                if response.status_code == 200:
                    text_result = response.json().get("transcription", "")
                    if text_result:
                        query = text_result
                        st.success(f"🗣️ You said: {text_result}")

    col1, col2 = st.columns([1, 1])
    with col1:
        process_clicked = st.button("⚡ Generate Both")
    with col2:
        clear_clicked = st.button("🧹 Clear Chat")

    if clear_clicked:
        st.session_state.conversation = []

    if process_clicked and query:
        st.info("🚀 Running Gemini and Groq models in parallel...")

        with st.spinner("Generating responses..."):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_gemini = executor.submit(lambda: llm_gemini.invoke(query).content)
                future_groq = executor.submit(query_groq, query)
                gemini_resp = future_gemini.result()
                groq_resp = future_groq.result()

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### 🤖 Gemini Response")
            st.markdown(gemini_resp)
        with col_b:
            st.markdown("### ⚡ Groq Response")
            st.markdown(groq_resp)

        chosen = st.radio("✅ Which response do you prefer?", ["Gemini", "Groq"], horizontal=True)
        if st.button("Confirm Choice"):
            st.success(f"You chose **{chosen}** response.")
            final_ans = gemini_resp if chosen == "Gemini" else groq_resp
            st.session_state.conversation.append((query, final_ans))

    if st.session_state.conversation:
        st.markdown("### 🗂️ Chat History")
        for user_q, assistant_a in st.session_state.conversation:
            st.markdown(f"**User:** {user_q}")
            st.code(assistant_a, language="markdown")
            st.markdown("---")

# =====================
# IMAGE GENERATOR TAB (unchanged)
# ==================
# =====================
# IMAGE GENERATOR TAB
# =====================
with tab2:
    st.subheader("🎨 Pollinations.AI Free Image Generator")

    img_prompt = st.text_input("📝 Enter your image prompt:", key="img_prompt")

    styles = ["Realistic", "Cartoon", "Fantasy", "Minimalist"]
    selected_style = st.radio("🎨 Choose a style:", styles, horizontal=True)

    def smart_enhance_prompt(user_prompt, style):
        quick_prompt = f"Rewrite this short prompt into a detailed {style} image generation description: {user_prompt}"
        response = llm.invoke(quick_prompt)
        return response.content.strip()

    @st.cache_data(show_spinner=False)
    def fetch_image(final_prompt, token):
        url = f"https://image.pollinations.ai/prompt/{final_prompt}?token={token}"
        return requests.get(url).content

    if st.button("Generate Image"):
        if not img_prompt:
            st.warning("⚠️ Please enter a prompt before generating an image.")
        else:
            with st.spinner(f"🎨 Generating {selected_style} image..."):
                final_prompt = smart_enhance_prompt(img_prompt, selected_style)
                try:
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

                content = [
                    {"type": "text", "text": qna_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]

                response_placeholder = st.empty()
                final_response = ""

                for chunk in llm.stream([HumanMessage(content=content)]):
                    final_response += chunk.content or ""
                    response_placeholder.markdown(f"**Answer (streaming):**\n\n{final_response}")
    # =========================
    # =========================
    # 📄 PDF Upload and OCR + Gemini Analysis
    # =========================
       # =========================
    # 📄 PDF Upload and Gemini Analysis (No Poppler Required)
    # =========================
    st.subheader("📄 Upload a PDF & Ask Gemini")

    uploaded_pdf = st.file_uploader("📂 Upload a PDF", type=["pdf"])
    pdf_question = st.text_input("💬 Ask something about the uploaded PDF:")

    if st.button("Analyze PDF"):
        if not uploaded_pdf:
            st.warning("⚠️ Please upload a PDF first.")
        elif not pdf_question:
            st.warning("⚠️ Please enter a question about the PDF.")
        else:
            with st.spinner("🔎 Reading and analyzing PDF..."):
                pdf_text = ""

                try:
                    # Try extracting text normally first
                    reader = PdfReader(uploaded_pdf)
                    for page in reader.pages:
                        pdf_text += page.extract_text() or ""
                except Exception as e:
                    st.error(f"PDF read error: {e}")

                # If no text found → fallback: treat each page as image
                if not pdf_text.strip():
                    st.info("🧠 No readable text found — converting pages using Pillow instead of Poppler...")
                    from PIL import Image
                    import fitz  # PyMuPDF (lightweight and Streamlit-safe)
                    uploaded_pdf.seek(0)
                    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")

                    for i, page in enumerate(doc):
                        pix = page.get_pixmap()
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        text = pytesseract.image_to_string(img)
                        pdf_text += f"\n--- Page {i+1} ---\n" + text

                if not pdf_text.strip():
                    st.error("⚠️ Could not extract text from PDF even after OCR.")
                else:
                    st.success("✅ Text extracted successfully! Sending to Gemini...")
                    content = [
                        {"type": "text", "text": f"Question: {pdf_question}\n\nPDF Content:\n{pdf_text[:8000]}"}
                    ]

                    response_placeholder = st.empty()
                    final_response = ""

                    for chunk in llm.stream([HumanMessage(content=content)]):
                        final_response += chunk.content or ""
                        response_placeholder.markdown(f"**Answer (streaming):**\n\n{final_response}") 



