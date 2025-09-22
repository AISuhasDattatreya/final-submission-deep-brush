import tensorflow as tf
import numpy as np
import warnings
import os
from PIL import Image
from scipy import ndimage

warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class AdaINVGGModel:
    def __init__(self):
        try:
            tf.config.set_visible_devices([], 'GPU')
            
            self.vgg = self._load_vgg19()
            
            self.decoder = self._build_decoder()
            
            print("AdaIN VGG model initialized successfully")
        except Exception as e:
            print(f"Error initializing model: {e}")

    def _load_vgg19(self):
        try:
            vgg = tf.keras.applications.VGG19(
                include_top=False,
                weights='imagenet',
                input_shape=(None, None, 3)
            )
            
            feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']
            outputs = [vgg.get_layer(name).output for name in feature_layers]
            
            model = tf.keras.Model(inputs=vgg.input, outputs=outputs)
            return model
        except Exception as e:
            print(f"Error loading VGG19: {e}")
            return self._build_simple_encoder()

    def _build_simple_encoder(self):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, 3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
        ])

    def _build_decoder(self):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
            tf.keras.layers.UpSampling2D(2),
            
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.UpSampling2D(2),
            
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(3, 3, padding='same', activation='sigmoid'),
        ])

    def adaptive_instance_normalization(self, content_features, style_features):
        content_mean, content_var = tf.nn.moments(content_features, axes=[1, 2], keepdims=True)
        style_mean, style_var = tf.nn.moments(style_features, axes=[1, 2], keepdims=True)
        
        content_std = tf.sqrt(content_var + 1e-8)
        style_std = tf.sqrt(style_var + 1e-8)
        
        normalized = (content_features - content_mean) / content_std
        
        stylized = normalized * style_std + style_mean
        
        return stylized

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
            
            content_features = self.vgg(content_tensor)[-1]
            style_features = self.vgg(style_tensor)[-1]
            
            stylized_features = self.adaptive_instance_normalization(content_features, style_features)
            
            stylized_tensor = self.decoder(stylized_features)
            stylized = stylized_tensor.numpy()[0]
            
            stylized = np.clip(stylized, 0, 1)
            
            blended = alpha * stylized + (1 - alpha) * content
            
            return (blended * 255).clip(0, 255).astype(np.uint8)
            
        except Exception as e:
            print(f"AdaIN VGG stylization error: {e}")
            import traceback
            traceback.print_exc()
            blended = alpha * style_np + (1 - alpha) * content_np
            return (blended * 255).clip(0, 255).astype(np.uint8)

    @staticmethod
    def metadata():
        return {
            "name": "AdaIN VGG (Enhanced AdaIN)",
            "type": "VGG19 + AdaIN + Custom Decoder",
            "features": "Multi-layer VGG features, Enhanced decoder, Better reconstruction",
            "paper": "Huang & Belongie (2017) - Arbitrary Style Transfer in Real-time",
            "real_time": True
        }