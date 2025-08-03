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
    model = get_model(model_name)
    metadata = model.metadata()
    ...
    return {
        "image_base64": encoded_img,
        "runtime_ms": round(runtime, 2),
        "model": metadata
    }