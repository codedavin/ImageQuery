# image_processor.py
from PIL import Image
from transformers import AutoProcessor, Idefics2ForConditionalGeneration
import torch
from config import DEVICE

def caption_image(image: Image.Image) -> str:
    try:
        processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
        model = Idefics2ForConditionalGeneration.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
        ).to(DEVICE)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Provide a detailed description of this image."}
                ]
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)

        generated_ids = model.generate(**inputs, max_new_tokens=500)
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error in captioning: {e}"