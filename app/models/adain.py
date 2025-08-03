import tensorflow as tf
import numpy as np
import warnings
import os
from PIL import Image

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class AdaINModel:
    def __init__(self):
        try:
            # Set TensorFlow to use CPU to avoid GPU-related issues
            tf.config.set_visible_devices([], 'GPU')
            print("AdaIN model initialized successfully")
        except Exception as e:
            print(f"Error initializing model: {e}")

    def stylize(self, content_np, style_np, alpha=1.0):
        """
        Applies a simplified stylization using basic image processing:
        Uses color blending and histogram matching techniques.
        """
        try:
            # Convert to float and normalize
            content = content_np.astype(np.float32) / 255.0
            style = style_np.astype(np.float32) / 255.0
            
            # Simple color transfer using RGB channels
            # Calculate mean and std for each channel
            content_mean = np.mean(content, axis=(0, 1))
            content_std = np.std(content, axis=(0, 1))
            style_mean = np.mean(style, axis=(0, 1))
            style_std = np.std(style, axis=(0, 1))
            
            # Apply color transfer to each channel
            stylized = np.zeros_like(content)
            for i in range(3):  # RGB channels
                if content_std[i] > 1e-8:
                    stylized[:, :, i] = (content[:, :, i] - content_mean[i]) * (style_std[i] / content_std[i]) + style_mean[i]
                else:
                    stylized[:, :, i] = content[:, :, i]
            
            # Clip to valid range
            stylized = np.clip(stylized, 0, 1)
            
            # Blend with original content
            blended = alpha * stylized + (1 - alpha) * content
            return (blended * 255).clip(0, 255).astype(np.uint8)
            
        except Exception as e:
            print(f"Stylization error: {e}")
            # Return simple blended content if stylization fails
            blended = alpha * style_np + (1 - alpha) * content_np
            return (blended * 255).clip(0, 255).astype(np.uint8)

    @staticmethod
    def metadata():
        return {
            "name": "AdaIN (Color Transfer)",
            "type": "Color Transfer",
            "features": "RGB color statistics transfer",
            "paper": "Simplified Color Transfer",
            "real_time": True
        }