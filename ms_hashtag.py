import streamlit as st
from PIL import Image, ImageEnhance
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import openai
import os

# Load environment variables from .env file
load_dotenv()

# OpenAI API credentials
openai.api_key = "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"

# Microsoft/beit-base-captioning model from Hugging Face
caption_model_name = "microsoft/git-base-coco"

# Load Microsoft/beit-base-captioning model
caption_tokenizer = AutoTokenizer.from_pretrained(caption_model_name)
caption_model = AutoModelForCausalLM.from_pretrained(caption_model_name)

# Generate caption for the uploaded image using Microsoft/beit-base-captioning model
def generate_caption(image):
    inputs = caption_tokenizer(image, return_tensors="pt")
    with torch.no_grad():
        caption_output = caption_model.generate(**inputs)
    caption = caption_tokenizer.decode(caption_output[0], skip_special_tokens=True)
    return caption

# Generate hashtags using OpenAI GPT-3.5 Turbo chat model
def generate_hashtags(caption):
    prompt = f"What are the best hashtags for the caption: '{caption}'?"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
    )
    hashtags = response.choices[0].text
    return hashtags.split("#")[1:]  # Split and remove the empty hashtag at the beginning

# Streamlit web application
def main():
    st.title("Image Caption and Hashtag Generator")

    # Image upload
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Generate caption
        caption = generate_caption(image)

        # Display caption
        st.subheader("Generated Caption:")
        st.write(caption)

        # Generate hashtags
        hashtags = generate_hashtags(caption)

        # Display hashtags
        st.subheader("Generated Hashtags:")
        for i, hashtag in enumerate(hashtags[:10]):
            st.write(f"{i+1}. #{hashtag.strip()}")

# Run the Streamlit web application
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()