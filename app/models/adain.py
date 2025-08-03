import tensorflow_hub as hub
import tensorflow as tf
import numpy as np

class AdaINModel:
    def __init__(self):
        self.model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

    def stylize(self, content_np, style_np, alpha=1.0):
        """
        Applies AdaIN (Adaptive Instance Normalization):
        Normalizes content features to match the mean and variance of style features.
        """
        content = tf.convert_to_tensor(content_np[np.newaxis, ...], dtype=tf.float32)
        style = tf.convert_to_tensor(style_np[np.newaxis, ...], dtype=tf.float32)
        stylized = self.model(content, style)[0][0].numpy()
        blended = alpha * stylized + (1 - alpha) * content_np
        return (blended * 255).clip(0, 255).astype(np.uint8)

    @staticmethod
    def metadata():
        return {
            "name": "AdaIN",
            "type": "Encoder-Decoder",
            "features": "1st-order statistics (mean, variance)",
            "paper": "Huang & Belongie (2017)",
            "real_time": True
        }