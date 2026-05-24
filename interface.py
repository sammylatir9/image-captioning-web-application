"""
INTERFACE MODULE
Creates a Gradio web interface that calls CaptionEngine.
"""

import gradio as gr
from engine import CaptionEngine

engine = CaptionEngine()

def run_caption(image):
    raw, refined = engine.caption_image(image)
    return raw, refined


app = gr.Interface(
    fn=run_caption,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Text(label="Raw Caption"), gr.Text(label="Refined Caption")],
    title="Image Captioning App",
    description="Uploads an image → Generates caption → Refines caption using Falcon LLM"
)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)