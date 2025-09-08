# Task Routing Gemini Chat Agent

A modular AI-powered chat agent that intelligently **routes user queries** to the best-suited Large Language Model (LLM) based on the task type (text or image).  
The agent classifies the incoming request, determines whether it’s an image or text task, and then uses the most appropriate LLM from a group of models to return the final response.

---

## 🚀 Features
- **Task Classification** – Automatically distinguishes between text and image-based tasks.  
- **Dynamic Model Routing** – Selects the best LLM for the given task type.  
- **Extensible Design** – Easily add new LLMs or extend with additional task types.  
- **Environment Secure** – Uses `.env` to store API keys securely.  

---

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sounakss7/Task-Routing-Gemini-Chat-Agent.git
   cd Task-Routing-Gemini-Chat-Agent
