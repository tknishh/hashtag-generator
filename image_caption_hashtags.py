import streamlit as st
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

# OpenAI GPT-3.5 Turbo chat model
chat_model = "gpt-3.5-turbo"

# OpenAI API credentials
openai.api_key = "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"

# Load the pre-trained image captioning model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Generate captions for the uploaded image
def generate_captions(image):
    # Preprocess the image
    image = Image.open(image).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")

    # Generate captions
    with torch.no_grad():
        outputs = model.generate(
            inputs.pixel_values,
            attention_mask=inputs.attention_mask,
            decoder_start_token_id=tokenizer.pad_token_id,
            max_length=128,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    # Decode the generated captions
    captions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return captions

# Generate hashtags using OpenAI GPT-3.5 Turbo chat model
def generate_hashtags(caption):
    prompt = "Generate 10 hashtags for this caption:\n" + caption
    response = openai.Completion.create(
        engine=chat_model,
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )
    hashtags = response.choices[0].text.strip().split("\n")
    return hashtags

# Streamlit web application
def main():
    st.title("Image Caption and Hashtag Generator")

    # Image upload
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = uploaded_image

        # Display uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Generate captions
        captions = generate_captions(image)

        # Display captions
        st.subheader("Generated Captions:")
        for i, caption in enumerate(captions):
            st.write(f"{i+1}. {caption}")

            # Generate hashtags
            hashtags = generate_hashtags(caption)

            # Display hashtags
            st.write("Generated Hashtags:")
            for i, hashtag in enumerate(hashtags[:10]):
                st.write(f"{i+1}. #{hashtag.strip()}")

# Run the Streamlit web application
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
