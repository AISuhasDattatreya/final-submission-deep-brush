import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error
import time

class StyleTransferEvaluator:
    
    def __init__(self):
        self.vgg = self._load_vgg19()
    
    def _load_vgg19(self):
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet',
            input_shape=(None, None, 3)
        )
        
        feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']
        outputs = [vgg.get_layer(name).output for name in feature_layers]
        
        return tf.keras.Model(inputs=vgg.input, outputs=outputs)
    
    def compute_feature_statistical_fidelity(self, content_np, style_np, stylized_np):
        content = self._preprocess_image(content_np)
        style = self._preprocess_image(style_np)
        stylized = self._preprocess_image(stylized_np)
        
        content_features = self.vgg(content)[-1]
        style_features = self.vgg(style)[-1]
        stylized_features = self.vgg(stylized)[-1]
        
        content_flat = content_features.numpy().flatten()
        stylized_flat = stylized_features.numpy().flatten()
        content_preservation = 1.0 - cosine(content_flat, stylized_flat)
        
        style_mean = tf.reduce_mean(style_features, axis=(1,2)).numpy()
        stylized_mean = tf.reduce_mean(stylized_features, axis=(1,2)).numpy()
        content_mean = tf.reduce_mean(content_features, axis=(1,2)).numpy()
        
        style_distance = np.linalg.norm(stylized_mean - style_mean)
        content_style_distance = np.linalg.norm(content_mean - style_mean)
        
        style_alignment = max(0, 1.0 - style_distance / (content_style_distance + 1e-8))
        
        fsf_score = 0.6 * content_preservation + 0.4 * style_alignment
        
        return {
            'fsf_score': fsf_score,
            'content_preservation': content_preservation,
            'style_alignment': style_alignment
        }
    
    def _preprocess_image(self, img_np):
        img = img_np.astype(np.float32) / 255.0
        return tf.convert_to_tensor(img[np.newaxis, ...])