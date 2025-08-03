import gradio as gr
import requests
import base64
from PIL import Image
import io

def stylize(content, style, alpha, model):
    files = {
        "content": ("content.png", content, "image/png"),
        "style": ("style.png", style, "image/png"),
    }
    data = {"alpha": alpha, "model_name": model}
    response = requests.post("http://localhost:8000/stylize", files=files, data=data)
    result = response.json()
    img_bytes = base64.b64decode(result["image_base64"])
    meta = result["model"]
    caption = f"{meta['name']} - {meta['features']}\\n({meta['type']}, Real-time: {meta['real_time']})"
    return Image.open(io.BytesIO(img_bytes)), caption

with gr.Blocks() as demo:
    gr.Markdown("# Deep Brush: AST Model Comparison Tool")
    with gr.Row():
        content = gr.Image(label="Content Image")
        style = gr.Image(label="Style Image")
    model_select = gr.Dropdown(["adain", "wct"], value="adain", label="Model")
    alpha = gr.Slider(0, 1, value=1.0, label="Style Blend Alpha")
    output_img = gr.Image(label="Stylized Output")
    output_info = gr.Textbox(label="Model Info")
    btn = gr.Button("Stylize")

    btn.click(fn=stylize, inputs=[content, style, alpha, model_select], outputs=[output_img, output_info])

demo.launch()