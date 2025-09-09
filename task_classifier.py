import requests
import streamlit as st

def handle_image_task(prompt: str):
    if not hf_api_key:
        return "⚠️ Hugging Face API key missing. Please add it to use image generation."

    api_url = "https://api-inference.huggingface.co/models/prompthero/openjourney"
    headers = {"Authorization": f"Bearer {hf_api_key}"}

    try:
        response = requests.post(api_url, headers=headers, json={"inputs": prompt})

        # If API returns error JSON
        if response.headers.get("content-type") == "application/json":
            error_msg = response.json()
            return f"⚠️ Hugging Face error: {error_msg.get('error', error_msg)}"

        # Otherwise, save image
        img_path = "generated.png"
        with open(img_path, "wb") as f:
            f.write(response.content)

        st.image(img_path, caption=f"Generated: {prompt}")
        return "✅ Image generated successfully!"
    except Exception as e:
        return f"⚠️ Error: {e}"
