"""
ENGINE MODULE
Loads BLIP for image captioning and GPT4All Falcon for caption refinement.
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from gpt4all import GPT4All
from config import BLIP_MODEL_PATH, GPT4ALL_MODEL_PATH


class CaptionEngine:
    def __init__(self):
        # Load BLIP processor & model
        self.processor = BlipProcessor.from_pretrained(BLIP_MODEL_PATH)
        self.model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_PATH)

        # Load GPT4All Falcon
        self.llm = GPT4All(GPT4ALL_MODEL_PATH)

    def generate_caption(self, image):
        """Generate raw caption using BLIP."""
        inputs = self.processor(image, return_tensors="pt")
        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)

    def refine_caption(self, caption):
        """Refine caption using GPT4All."""
        prompt = f"Make this caption clearer: {caption}"
        return self.llm.generate(prompt, max_tokens=60)

    def caption_image(self, image):
        raw = self.generate_caption(image)
        refined = self.refine_caption(raw)
        return raw, refined
