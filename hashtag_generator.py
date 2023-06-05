import streamlit as st
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import openai
import os, re

# Load environment variables
load_dotenv()

# Set up OpenAI API
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = "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"

# Set up Image Captioning Model
IMAGE_CAPTION_MODEL_NAME = "microsoft/git-base"
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

# Function to preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Resize image to match the input size of the model

    return image

# Function to generate image caption
def generate_caption(image_path):
    image = preprocess_image(image_path)
    inputs = tokenizer_caption.encode_plus("", return_tensors="pt", padding=True, truncation=True)

    # Generate the caption
    output = model_caption.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    caption = tokenizer_caption.decode(output[0], skip_special_tokens=True)

    return caption

# Function to generate hashtags using OpenAI GPT-3.5 Turbo
# Function to generate hashtags using OpenAI GPT-3.5 Turbo
def generate_hashtags(caption):
    prompt = f"Generate 10 engaging hashtags for the image caption: {caption}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        n=10,
        temperature=0.7,
        stop=None
    )

    hashtags = re.findall(r"#\w+", response.choices[0].text)

    return hashtags

# Main function
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
