import gradio as gr
import requests
import base64
from PIL import Image
import io
import numpy as np

def stylize(content, style, alpha, model):
    try:
        # Convert Gradio image data to PIL Image
        if content is None or style is None:
            return None, "Please upload both content and style images"
        
        print(f"Content type: {type(content)}")
        print(f"Style type: {type(style)}")
        
        # Handle numpy arrays from Gradio
        if isinstance(content, np.ndarray):
            print(f"Content array shape: {content.shape}")
            content_img = Image.fromarray(content)
        elif isinstance(content, dict):
            print(f"Content dict keys: {content.keys()}")
            content_img = Image.open(content['name'])
        else:
            content_img = Image.open(content)
            
        if isinstance(style, np.ndarray):
            print(f"Style array shape: {style.shape}")
            style_img = Image.fromarray(style)
        elif isinstance(style, dict):
            print(f"Style dict keys: {style.keys()}")
            style_img = Image.open(style['name'])
        else:
            style_img = Image.open(style)
        
        print(f"Content image size: {content_img.size}")
        print(f"Style image size: {style_img.size}")
        
        # Convert to RGB if necessary
        if content_img.mode != 'RGB':
            content_img = content_img.convert('RGB')
        if style_img.mode != 'RGB':
            style_img = style_img.convert('RGB')
        
        # Save to bytes for API
        content_buffer = io.BytesIO()
        style_buffer = io.BytesIO()
        content_img.save(content_buffer, format='PNG')
        style_img.save(style_buffer, format='PNG')
        
        print(f"Content buffer size: {len(content_buffer.getvalue())}")
        print(f"Style buffer size: {len(style_buffer.getvalue())}")
        
        files = {
            "content": ("content.png", content_buffer.getvalue(), "image/png"),
            "style": ("style.png", style_buffer.getvalue(), "image/png"),
        }
        data = {"alpha": alpha, "model_name": model}
        
        response = requests.post("http://localhost:8001/stylize", files=files, data=data)
        result = response.json()
        
        if "error" in result:
            return None, f"Error: {result['error']}"
        
        img_bytes = base64.b64decode(result["image_base64"])
        meta = result["model"]
        caption = f"{meta['name']} - {meta['features']}\\n({meta['type']}, Real-time: {meta['real_time']})"
        return Image.open(io.BytesIO(img_bytes)), caption
        
    except Exception as e:
        print(f"Interface error: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# Deep Brush: AST Model Comparison Tool")
    with gr.Row():
        content = gr.Image(label="Content Image")
        style = gr.Image(label="Style Image")
    model_select = gr.Dropdown(["adain", "adain_vgg", "wct"], value="adain", label="Model")
    alpha = gr.Slider(0, 1, value=1.0, label="Style Blend Alpha")
    output_img = gr.Image(label="Stylized Output")
    output_info = gr.Textbox(label="Model Info")
    btn = gr.Button("Stylize")

    btn.click(fn=stylize, inputs=[content, style, alpha, model_select], outputs=[output_img, output_info])

demo.launch()