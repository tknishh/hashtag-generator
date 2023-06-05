import torch
from PIL import Image
import requests
from transformers import T5ForConditionalGeneration, T5Tokenizer


def generate_caption(image_path):
    model = T5ForConditionalGeneration.from_pretrained("mrm8488/t5-base-finetuned-caption-generator")
    tokenizer = T5Tokenizer.from_pretrained("mrm8488/t5-base-finetuned-caption-generator")

    image = Image.open(image_path)
    image = image.convert("RGB")
    image_tensor = torch.tensor([torchvision.transforms.functional.to_tensor(image)])

    input_ids = tokenizer.encode("generate a caption for the image:", return_tensors="pt")
    input_ids = torch.cat([input_ids, image_tensor], dim=1)

    output = model.generate(input_ids=input_ids)
    caption = tokenizer.decode(output[0], skip_special_tokens=True)

    return caption


def generate_hashtags(description):
    description = description.lower()
    hashtags = ['#' + word for word in description.split()]
    top_10_hashtags = hashtags[:10]
    return top_10_hashtags


def main():
    image_path = "levi-arnold-AkoHPGMV4X8-unsplash.jpg"
    caption = generate_caption(image_path)
    hashtags = generate_hashtags(caption)

    print("Caption:")
    print(caption)
    print("\nHashtags:")
    print(hashtags)


if __name__ == "__main__":
    main()
