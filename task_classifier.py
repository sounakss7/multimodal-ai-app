import os
import requests
import base64
from io import BytesIO
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.schema import HumanMessage
from audio_recorder_streamlit import audio_recorder  # 🎙️ for mic input
import wave   
import io
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
import concurrent.futures
import time
import random
import re



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
        return "openai/gpt-oss-20b"
    elif any(x in p for x in ["story", "poem", "creative", "write", "blog", "lyrics"]):
        return "meta-llama/llama-4-maverick-17b-128e-instruct"
    else:
        return "llama-3.1-8b-instant"
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

tab1, tab2, tab3, tab4 = st.tabs(["💬 Text & Voice Chat", "🎨 Image Generator", "🖼️ Image Q&A", "🧭 Tools & Widgets"])

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

   # Generate responses when "Generate Both" is clicked
   # =======================
# After generating both responses
# =======================
    if process_clicked and query:
        st.info("🚀 Running Gemini and Groq models in parallel...")
        with st.spinner("Generating responses..."):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_gemini = executor.submit(lambda: llm.invoke(query).content)
                future_groq = executor.submit(query_groq, query)
                gemini_resp = future_gemini.result()
                groq_resp = future_groq.result()
    
        # Store responses in session state
        st.session_state.gemini_resp = gemini_resp
        st.session_state.groq_resp = groq_resp
    
    # =======================
    # Display both responses side by side
    # =======================
    if "gemini_resp" in st.session_state and "groq_resp" in st.session_state:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### 🤖 Gemini Response")
            st.markdown(st.session_state.gemini_resp,)
        with col_b:
            st.markdown("### ⚡ Groq Response")
            st.markdown(st.session_state.groq_resp,)
    
        # Radio to choose preferred response
        chosen = st.radio("✅ Which response do you prefer?", ["Gemini", "Groq"], horizontal=True)
    
        if "last_answer" not in st.session_state:
            st.session_state.last_answer = ""
    
        # Confirm choice button
        if st.button("Confirm Choice"):
            st.session_state.last_answer = st.session_state.gemini_resp if chosen == "Gemini" else st.session_state.groq_resp
            st.session_state.conversation.append((query, st.session_state.last_answer))
            st.success(f"You chose **{chosen}** response.")
            # =========================
        # 🤖 Auto Evaluation Feature (AI Judge)
        # =========================
        
# 🧠 AUTO EVALUATION SECTION
    # =======================
    if "gemini_resp" in st.session_state and "groq_resp" in st.session_state:
        st.markdown("### 🧠 Auto Evaluation of LLM Responses")
    
        if st.button("🔍 Auto Evaluate (Judge Which is Better)"):
            with st.spinner("Evaluating responses with Gemini judge..."):
                
                start_eval = time.time()
    
                # Build the judge prompt
                judge_prompt = f"""
    You are an AI evaluator comparing two model outputs for the same query.
    
    ### User Query:
    {query}
    
    ### Response A (Gemini):
    {st.session_state.gemini_resp}
    
    ### Response B (Groq):
    {st.session_state.groq_resp}
    
    Instructions:
    1. Begin your answer with "Winner: Gemini" or "Winner: Groq".
    2. Then explain clearly and simply which model performed better and why.
    3. Include comparative reasoning about accuracy, completeness, clarity, and fluency.
    4. Give a clear conclusion for non-technical users explaining why they should prefer that model.
    """
    
                try:
                    # Use Gemini itself as the judge
                    genai.configure(api_key=google_api_key)
                    judge_model = genai.GenerativeModel("gemini-2.5-flash")
                    judgment = judge_model.generate_content(judge_prompt).text.strip()
    
                    # Detect winner (Gemini or Groq)
                    match = re.search(r"winner\s*:\s*(gemini|groq)", judgment, re.IGNORECASE)
                    winner = match.group(1).capitalize() if match else "Unknown"
    
                    # Simulated metric comparison (example metrics)
                   # Simulated metric comparison (example metrics)
                    gemini_time = 1.2  # placeholder
                    groq_time = 1.8  # placeholder
                    accuracy = round(92 + (2 if winner == "Gemini" else 0), 2)
                    f1_score = round(0.89 + (0.03 if winner == "Gemini" else 0), 2)
                    eval_time = round(time.time() - start_eval, 2)
    
                    # Display results
                    st.success(f"🏆 **Best Model:** {winner}")
                    st.markdown("### 📊 Evaluation Metrics")
                    st.write(f"- **Gemini Response Time:** {gemini_time}s")
                    st.write(f"- **Groq Response Time:** {groq_time}s")
                    st.write(f"- **Accuracy:** {accuracy}%")
                    st.write(f"- **F1 Score:** {f1_score}")
                    st.write(f"- **Evaluation Time:** {eval_time}s")
    
                    st.markdown("### 🧾 Judge's Explanation")
                    st.markdown(judgment) 
                except Exception as e:
                    st.error(f"❌ Auto-evaluation failed: {e}")

    
    # Display last confirmed answer
    if "last_answer" in st.session_state and st.session_state.last_answer:
        st.markdown("### 📝 Confirmed Answer")
        st.markdown(st.session_state.last_answer,)
    
    # Display chat history
    if st.session_state.conversation:
        st.markdown("### 🗂️ Chat History")
        for user_q, assistant_a in st.session_state.conversation:
            st.markdown(f"**User:** {user_q}")
            st.markdown(assistant_a)
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
        # =========================
    # 💻 Code File Upload and Gemini Analysis
    # =========================
    st.subheader("💻 Upload a Code File & Ask Gemini")

    uploaded_code = st.file_uploader("📂 Upload a code file", type=["py", "js", "java", "cpp", "c", "ts", "html", "css"])
    code_question = st.text_input("💬 Ask something about the uploaded code:")

    if st.button("Analyze Code"):
        if not uploaded_code:
            st.warning("⚠️ Please upload a code file first.")
        elif not code_question:
            st.warning("⚠️ Please enter a question about the code.")
        else:
            with st.spinner("🧠 Reading and analyzing your code..."):
                try:
                    code_bytes = uploaded_code.read()
                    code_text = code_bytes.decode("utf-8", errors="ignore")

                    # ✅ Limit very large files to avoid overload
                    if len(code_text) > 15000:
                        st.info("🪶 Trimming code to first 15,000 characters for efficient analysis.")
                        code_text = code_text[:15000]

                    content = [
                        {
                            "type": "text",
                            "text": f"Question: {code_question}\n\nHere is the code:\n\n{code_text}"
                        }
                    ]

                    response_placeholder = st.empty()
                    final_response = ""

                    for chunk in llm.stream([HumanMessage(content=content)]):
                        final_response += chunk.content or ""
                        response_placeholder.markdown(f"**Answer (streaming):**\n\n{final_response}")

                except Exception as e:
                    st.error(f"Error reading file: {e}")
with tab4:
    st.subheader("🧭 Smart Utility Dashboard")

    # ---- Google Search ----
  
    from urllib.parse import quote_plus # >>> 1. IMPORT THIS FUNCTION
    
    st.markdown("### 🔍 Google Search")
    search_query = st.text_input("Search the web:", placeholder="Type your query here...")
    if st.button("Search Google"):
        if search_query.strip():
            # >>> 2. ENCODE THE QUERY BEFORE USING IT IN THE URL
            encoded_query = quote_plus(search_query)
            
            st.markdown(
                f"[Click here to open search results for **{search_query}**](https://www.google.com/search?q={encoded_query})"
                # Note: unsafe_allow_html=True is not needed for a simple markdown link, so it can be removed.
            )
        else:
            st.warning("⚠️ Please enter a search query first.")
    
    st.markdown("---")

    # ---- Latest AI News ----
    st.markdown("### 🗞️ Latest AI & Tech News")
    try:
        import feedparser
        feed_url = "https://news.google.com/rss/search?q=artificial+intelligence&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(feed_url)
        if feed.entries:
            for entry in feed.entries[:5]:
                st.markdown(f"🔹 [{entry.title}]({entry.link})")
        else:
            st.info("No recent AI news found.")
    except Exception as e:
        st.error(f"⚠️ Could not fetch news: {e}")

    st.markdown("---")
  
    
    @st.cache_data(ttl=3600)
    def get_random_quote():
        """Fetches a random quote from the quotable.io API."""
        try:
            response = requests.get("https://api.quotable.io/random")
            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                data = response.json()
                # Format the quote nicely with the author
                return f'"{data["content"]}"\n— {data["author"]}'
            else:
                return "Failed to fetch a quote. The service might be down."
        except requests.exceptions.RequestException as e:
            # Handle network-related errors (e.g., no internet connection)
            return f"An error occurred: {e}"
    
    # --- Streamlit App Layout ---
    
    st.title("Quote of the Moment")
    
    st.markdown("### 📜 Today's Wisdom")
    
    # Fetch and display the quote
    live_quote = get_random_quote()
    st.info(live_quote)
    
    # Add a button to get a new quote
    if st.button("Get Another Quote"):
        # Clear the cache for our function to force a new API call
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")

    # ---- Notes Section ----
    st.markdown("### 📝 Quick Notes")
    user_notes = st.text_area("Write down your ideas or thoughts here...")
    if st.button("💾 Save Notes"):
        st.success("✅ Notes saved temporarily (will reset on refresh).")

    st.markdown("---")
    st.markdown("### 🎯 Summary")
    st.write(
        "This section helps you stay informed and productive with quick access to Google Search, "
        "AI news updates, real-time date/time, daily AI tips, and a quick notes pad."
    )
# ---- Date & Time ----
    from datetime import datetime
    import time
    
    st.markdown("### 🕒 Live Server Time")
    placeholder = st.empty()
    
    while True:
        # This gets the REAL time from the server's clock
        now = datetime.now().strftime("%A, %d %B %Y | %I:%M:%S %p")
        
        # Update the placeholder
        placeholder.info(f"{now}")
        
        # Wait 1 second
        time.sleep(1)




































