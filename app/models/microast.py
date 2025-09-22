import tensorflow as tf
import numpy as np
import warnings
import os
from PIL import Image
from scipy import ndimage

warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class MicroASTModel:
    
    def __init__(self):
        try:
            tf.config.set_visible_devices([], 'GPU')
            
            self.content_encoder = self._build_content_encoder()
            self.style_encoder = self._build_style_encoder()
            self.decoder = self._build_micro_decoder()
            
            print("MicroAST model initialized successfully with micro encoders and decoder")
        except Exception as e:
            print(f"Error initializing MicroAST model: {e}")

    def _build_content_encoder(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 7, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
        ], name='micro_content_encoder')
        
        return model

    def _build_style_encoder(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 7, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='linear'),
        ], name='micro_style_encoder')
        
        return model

    def _build_micro_decoder(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Conv2DTranspose(16, 7, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(3, 3, padding='same', activation='sigmoid'),
        ], name='micro_decoder')
        
        return model

    def _adaptive_instance_normalization(self, content_features, style_features):
        content_mean, content_var = tf.nn.moments(content_features, axes=[1, 2], keepdims=True)
        style_mean, style_var = tf.nn.moments(style_features, axes=[1, 2], keepdims=True)
        
        content_normalized = (content_features - content_mean) / tf.sqrt(content_var + 1e-8)
        
        stylized = content_normalized * tf.sqrt(style_var + 1e-8) + style_mean
        
        return stylized

    def _extract_style_statistics(self, style_image):
        style_features = self.style_encoder(style_image)
        
        batch_size = tf.shape(style_features)[0]
        style_mean = style_features[:, :128]
        style_std = style_features[:, 128:]
        
        style_std = tf.nn.softplus(style_std) + 1e-8
        
        return style_mean, style_std

    def _apply_style_modulation(self, content_features, style_mean, style_std):
        feature_shape = tf.shape(content_features)
        batch_size = feature_shape[0]
        height = feature_shape[1]
        width = feature_shape[2]
        channels = feature_shape[3]
        
        style_mean = tf.reshape(style_mean, [batch_size, 1, 1, channels])
        style_std = tf.reshape(style_std, [batch_size, 1, 1, channels])
        
        modulated_features = content_features * style_std + style_mean
        
        return modulated_features

    def stylize(self, content_np, style_np, alpha=1.0):
        try:
            content = content_np.astype(np.float32) / 255.0
            style = style_np.astype(np.float32) / 255.0
            
            if content.shape != style.shape:
                style_resized = ndimage.zoom(style, (content.shape[0]/style.shape[0], 
                                                    content.shape[1]/style.shape[1], 1), order=1)
                style = style_resized
            
            content_tensor = tf.convert_to_tensor(content[np.newaxis, ...])
            style_tensor = tf.convert_to_tensor(style[np.newaxis, ...])
            
            content_features = self.content_encoder(content_tensor)
            
            style_mean, style_std = self._extract_style_statistics(style_tensor)
            
            stylized_features = self._apply_style_modulation(content_features, style_mean, style_std)
            
            stylized_tensor = self.decoder(stylized_features)
            stylized = stylized_tensor.numpy()[0]
            
            stylized = np.clip(stylized, 0, 1)
            
            blended = alpha * stylized + (1 - alpha) * content
            
            return (blended * 255).clip(0, 255).astype(np.uint8)
            
        except Exception as e:
            print(f"MicroAST stylization error: {e}")
            import traceback
            traceback.print_exc()
            blended = alpha * style_np + (1 - alpha) * content_np
            return (blended * 255).clip(0, 255).astype(np.uint8)

    @staticmethod
    def metadata():
        return {
            "name": "MicroAST (Ultra-Fast Style Transfer)",
            "type": "Micro Encoders + Decoder",
            "features": "Lightweight, Ultra-fast, 4K capable, Micro architecture",
            "paper": "MicroAST: Ultra-Fast and Ultra-Resolution Arbitrary Style Transfer (2022)",
            "real_time": True
        }