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
        
        # Handle numpy arrays from Gradio 5.x
        if isinstance(content, np.ndarray):
            print(f"Content array shape: {content.shape}")
            content_img = Image.fromarray(content)
        else:
            content_img = Image.open(content)
            
        if isinstance(style, np.ndarray):
            print(f"Style array shape: {style.shape}")
            style_img = Image.fromarray(style)
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

# Create Gradio 5.x interface
with gr.Blocks(title="Deep Brush: AST Model Comparison Tool") as demo:
    gr.Markdown("# Deep Brush: AST Model Comparison Tool")
    gr.Markdown("Upload a content image and a style image to see different neural style transfer algorithms in action!")
    
    with gr.Row():
        with gr.Column():
            content = gr.Image(label="Content Image", type="pil")
            style = gr.Image(label="Style Image", type="pil")
        
        with gr.Column():
            model_select = gr.Dropdown(
                choices=["adain", "adain_vgg", "wct", "ast"], 
                value="adain", 
                label="Model",
                info="Choose the style transfer algorithm"
            )
            alpha = gr.Slider(
                minimum=0, 
                maximum=1, 
                value=1.0, 
                step=0.1,
                label="Style Blend Alpha",
                info="Higher values = more style, lower values = more content"
            )
            btn = gr.Button("Stylize", variant="primary")
    
    with gr.Row():
        output_img = gr.Image(label="Stylized Output", type="pil")
        output_info = gr.Textbox(label="Model Info", lines=3)

    btn.click(
        fn=stylize, 
        inputs=[content, style, alpha, model_select], 
        outputs=[output_img, output_info]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)