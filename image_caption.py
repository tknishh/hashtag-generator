import streamlit as st
from PIL import Image, ImageEnhance
from transformers import AutoProcessor, AutoModelForCausalLM
from dotenv import load_dotenv
import openai
import os

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API credentials
openai.api_key = "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"


# Set up the model and processor
processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
# Initialize the pre-trained model and processor for image captioning

# Function to generate image captions
def generate_image_captions(image_files):
    text_descriptions = []
    for image_file in image_files:
        # Open the image
        image = Image.open(image_file)

        # Convert the image to RGB mode if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Enhance the image by increasing sharpness
        enhanced_image = ImageEnhance.Sharpness(image).enhance(2.0)

        # Convert the enhanced image to pixel values using the processor
        pixel_values = processor(images=enhanced_image, return_tensors="pt").pixel_values

        # Generate captions using the model
        generated_ids = model.generate(pixel_values=pixel_values, max_length=50, num_return_sequences=1)
        generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Append the caption to the list of text descriptions
        text_descriptions.append(generated_captions[0])

    return text_descriptions

# Function to get final caption using OpenAI API
def get_final_caption(text, num_captions=1):
    # Use OpenAI's ChatCompletion API to generate final captions
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"You are a social media manager expert who is an expert in writing viral social media captions with a minimum of 10 words and a maximum of 20 words. Please use this vague description of my image: {text}, and make sure that the caption is relevant and compelling."
            }
        ],
        temperature=0.7,
        n=num_captions
    )

    return response

# Streamlit app
def main():
    st.title("Image Caption Generator")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.read())

        # Display uploaded image
        image = Image.open(temp_image_path)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # User input for number of captions
        num_captions = st.number_input("Enter the number of captions", value=1, min_value=1, max_value=5, step=1)

        # Generate caption
        text_descriptions = generate_image_captions([temp_image_path])
        caption_objects = [get_final_caption(description, num_captions) for description in text_descriptions]

        # Print the captions
        for i, caption_object in enumerate(caption_objects):
            st.subheader(f"Image {i+1} Captions:")
            choices = caption_object['choices']
            for j, choice in enumerate(choices):
                caption = choice['message']['content']
                st.write(f"Caption {j+1}: {caption}")
            st.write()

        # Remove the temporary image file
        os.remove(temp_image_path)

if __name__ == "__main__":
    main()