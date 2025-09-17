# 🤖 Multimodal AI App (Text + Image Generator + Image Q&A)
Check out the live app here: [AgentAI007](https://agentai007.streamlit.app/)



A **Streamlit-based multimodal AI application** that combines the power of **Google Gemini** (for text + multimodal tasks) and **Pollinations.AI** (for free image generation).  

This app supports:  
- 💬 **Text Chat** with Gemini (conversation memory + streaming responses)  
- 🎨 **AI Image Generator** with Gemini-enhanced prompts + Pollinations API  
- 🖼️ **Image Q&A** where you can upload an image and ask Gemini questions about it  

---

## 🚀 Features

### 💬 Text Chat
- Chat with Gemini like a smart assistant  
- Maintains conversation history across turns  
- Streams responses in real time  
- Special command: asks *"Who built this?"* → replies with **Sounak Sarkar**  

### 🎨 Image Generator
- Enter a text prompt + choose a style (Realistic, Cartoon, Fantasy, Minimalist)  
- Gemini automatically **enhances your prompt** for better results  
- Uses **Pollinations.AI** for free high-quality image generation  
- Download generated images as PNG  

### 🖼️ Image Q&A
- Upload an image (`.jpg`, `.jpeg`, `.png`)  
- Ask Gemini any question about the image  
- Works via **multimodal input** (text + base64 image)  
- Streams Gemini’s analysis in real time  

---

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/) – Web UI framework  
- [Google Gemini API](https://ai.google.dev/) – Multimodal AI model  
- [Pollinations.AI](https://pollinations.ai/) – Free image generation API  
- [LangChain](https://www.langchain.com/) – Wrapper for LLMs  
- [Python-dotenv](https://pypi.org/project/python-dotenv/) – Load environment variables  
- [Pillow (PIL)](https://pypi.org/project/Pillow/) – Image handling  

---

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/multimodal-ai-app.git
cd multimodal-ai-app

### 2. Create and activate a virtual environment

```bash
# Create a virtual environment
python -m venv venv

# Activate on Mac/Linux
source venv/bin/activate

# Activate on Windows (PowerShell or CMD)
venv\Scripts\activate


