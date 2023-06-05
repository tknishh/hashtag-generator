import streamlit as st
from PIL import Image, ImageEnhance
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import openai
import os

# Load environment variables
load_dotenv()

# Set up OpenAI API
OPENAI_API_KEY = os.getenv("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")
openai.api_key = OPENAI_API_KEY

# Set up Image Captioning Model
IMAGE_CAPTION_MODEL_NAME = "microsoft/git-base-coco"
tokenizer_caption = AutoTokenizer.from_pretrained(IMAGE_CAPTION_MODEL_NAME)
model_caption = AutoModelForCausalLM.from_pretrained(IMAGE_CAPTION_MODEL_NAME)

# Streamlit Application
def main():
    st.title("Image Hashtag Generator")

    uploaded_files = st.file_uploader("Upload image(s)", accept_multiple_files=True, type=["png", "jpg", "jpeg"])
    if uploaded_files:
        for file in uploaded_files:
            image = Image.open(file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            if st.button("Generate Hashtags"):
                caption = generate_caption(file)
                st.subheader("Generated Caption:")
                st.write(caption)

                hashtags = generate_hashtags(caption)
                st.subheader("Generated Hashtags:")
                for i, hashtag in enumerate(hashtags, 1):
                    st.write(f"{i}. {hashtag}")
                st.markdown("---")

# Function to generate image caption
def generate_caption(image_path):
    image = Image.open(image_path)

    input_text = "generate a caption for the image:"
    inputs = tokenizer_caption(image_path, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids

    output = model_caption.generate(input_ids=input_ids)
    caption = tokenizer_caption.decode(output[0], skip_special_tokens=True)
    
    return caption

# Function to generate hashtags using OpenAI GPT-3.5 Turbo
def generate_hashtags(caption):
    prompt = f"Generate 10 engaging hashtags for the image caption: {caption}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        n=10,
        temperature=0.7,
        stop=None,
        log_level="info"
    )
    
    hashtags = [hashtag['choices'][0]['text'].strip() for hashtag in response['choices']]
    
    return hashtags

# Main function
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
