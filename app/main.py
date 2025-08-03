from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from app.models.registry import get_model
from PIL import Image
from io import BytesIO
import numpy as np
import time
import base64
from app.models.registry import get_model


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def root():
    return {"message": "Deep Brush AST API"}



@app.post("/stylize")
async def stylize_image(
    content: UploadFile = File(...),
    style: UploadFile = File(...),
    alpha: float = Form(1.0),
    model_name: str = Form("adain")
):
    start_time = time.time()
    
    try:
        # Load images with error handling
        content_data = await content.read()
        style_data = await style.read()
        
        print(f"Content file size: {len(content_data)} bytes")
        print(f"Style file size: {len(style_data)} bytes")
        print(f"Content filename: {content.filename}")
        print(f"Style filename: {style.filename}")
        print(f"Content content_type: {content.content_type}")
        print(f"Style content_type: {style.content_type}")
        
        # Check first few bytes to see what we're dealing with
        print(f"Content first 20 bytes: {content_data[:20]}")
        print(f"Style first 20 bytes: {style_data[:20]}")
        
        # Try to identify the image format
        try:
            content_img = Image.open(BytesIO(content_data))
            print(f"Content image format: {content_img.format}")
            print(f"Content image mode: {content_img.mode}")
            print(f"Content image size: {content_img.size}")
        except Exception as e:
            print(f"Error opening content image: {e}")
            # Try to save and reload
            temp_buffer = BytesIO(content_data)
            content_img = Image.open(temp_buffer)
        
        try:
            style_img = Image.open(BytesIO(style_data))
            print(f"Style image format: {style_img.format}")
            print(f"Style image mode: {style_img.mode}")
            print(f"Style image size: {style_img.size}")
        except Exception as e:
            print(f"Error opening style image: {e}")
            # Try to save and reload
            temp_buffer = BytesIO(style_data)
            style_img = Image.open(temp_buffer)
        
        # Convert to RGB if necessary
        if content_img.mode != 'RGB':
            content_img = content_img.convert('RGB')
        if style_img.mode != 'RGB':
            style_img = style_img.convert('RGB')
        
        # Convert to numpy arrays
        content_np = np.array(content_img)
        style_np = np.array(style_img)
        
        print(f"Content image shape: {content_np.shape}")
        print(f"Style image shape: {style_np.shape}")
        
        # Ensure both images have the same size
        if content_np.shape != style_np.shape:
            style_img = style_img.resize(content_img.size)
            style_np = np.array(style_img)
            print(f"Resized style image shape: {style_np.shape}")
        
        # Get model and stylize
        model = get_model(model_name)
        stylized_np = model.stylize(content_np, style_np, alpha)
        
        # Convert back to PIL Image
        stylized_img = Image.fromarray(stylized_np)
        
        # Encode to base64
        buffer = BytesIO()
        stylized_img.save(buffer, format="PNG")
        encoded_img = base64.b64encode(buffer.getvalue()).decode()
        
        runtime = (time.time() - start_time) * 1000
        metadata = model.metadata()
        
        return {
            "image_base64": encoded_img,
            "runtime_ms": round(runtime, 2),
            "model": metadata
        }
        
    except Exception as e:
        print(f"Error in stylize_image: {e}")
        import traceback
        traceback.print_exc()
        # Return a simple error response
        return {
            "error": f"Image processing failed: {str(e)}",
            "runtime_ms": 0,
            "model": {"name": "Error", "type": "Error", "features": "Error", "real_time": False}
        }