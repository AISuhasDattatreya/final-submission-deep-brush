import tensorflow as tf
import numpy as np
import warnings
import os
from PIL import Image
from scipy import ndimage
from scipy.linalg import sqrtm

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ASTModel:
    def __init__(self):
        try:
            # Set TensorFlow to use CPU to avoid GPU-related issues
            tf.config.set_visible_devices([], 'GPU')
            
            # Load pre-trained VGG19 for feature extraction
            self.vgg = self._load_vgg19()
            
            # Build decoder for reconstruction
            self.decoder = self._build_decoder()
            
            print("AST model initialized successfully with VGG19 encoder-decoder")
        except Exception as e:
            print(f"Error initializing AST model: {e}")

    def _load_vgg19(self):
        """
        Load pre-trained VGG19 model for feature extraction
        """
        try:
            # Load VGG19 without top layers
            vgg = tf.keras.applications.VGG19(
                include_top=False,
                weights='imagenet',
                input_shape=(None, None, 3)
            )
            
            # Create a model that outputs features from multiple layers
            # AST uses features from relu4_1 layer
            feature_layer = 'block4_conv1'
            model = tf.keras.Model(inputs=vgg.input, outputs=vgg.get_layer(feature_layer).output)
            return model
        except Exception as e:
            print(f"Error loading VGG19: {e}")
            return None

    def _build_decoder(self):
        """
        Build decoder for image reconstruction from features
        """
        if not TENSORFLOW_AVAILABLE:
            return None
            
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
            tf.keras.layers.UpSampling2D(2),
            
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.UpSampling2D(2),
            
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.UpSampling2D(2),
            
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(3, 3, padding='same', activation='sigmoid'),
        ])
        return model

    def whitening_and_coloring_transform(self, content_features, style_features):
        """
        Apply Whitening and Coloring Transform (WCT) for AST
        This is the core of Arbitrary Style Transfer
        """
        try:
            # Flatten spatial dimensions
            content_flat = tf.reshape(content_features, [-1, tf.shape(content_features)[-1]])
            style_flat = tf.reshape(style_features, [-1, tf.shape(style_features)[-1]])
            
            # Compute mean and covariance
            content_mean = tf.reduce_mean(content_flat, axis=0)
            style_mean = tf.reduce_mean(style_flat, axis=0)
            
            content_centered = content_flat - content_mean
            style_centered = style_flat - style_mean
            
            # Compute covariance matrices
            content_cov = tf.matmul(content_centered, content_centered, transpose_a=True) / (tf.cast(tf.shape(content_flat)[0], tf.float32) - 1)
            style_cov = tf.matmul(style_centered, style_centered, transpose_a=True) / (tf.cast(tf.shape(style_flat)[0], tf.float32) - 1)
            
            # Eigenvalue decomposition for whitening
            content_eigenvals, content_eigenvecs = tf.linalg.eigh(content_cov)
            style_eigenvals, style_eigenvecs = tf.linalg.eigh(style_cov)
            
            # Add small epsilon to avoid division by zero
            epsilon = 1e-8
            content_eigenvals = tf.maximum(content_eigenvals, epsilon)
            style_eigenvals = tf.maximum(style_eigenvals, epsilon)
            
            # Whitening matrix
            whitening = tf.matmul(tf.matmul(content_eigenvecs, tf.linalg.diag(1.0 / tf.sqrt(content_eigenvals))), content_eigenvecs, transpose_b=True)
            
            # Coloring matrix
            coloring = tf.matmul(tf.matmul(style_eigenvecs, tf.linalg.diag(tf.sqrt(style_eigenvals))), style_eigenvecs, transpose_b=True)
            
            # Apply WCT
            whitened = tf.matmul(content_centered, whitening)
            colored = tf.matmul(whitened, coloring)
            
            # Add style mean
            transformed = colored + style_mean
            
            # Reshape back to original spatial dimensions
            original_shape = tf.shape(content_features)
            transformed_features = tf.reshape(transformed, original_shape)
            
            return transformed_features
            
        except Exception as e:
            print(f"WCT error: {e}")
            return content_features

    def stylize(self, content_np, style_np, alpha=1.0):
        """
        Arbitrary Style Transfer using WCT
        """
        try:
            # Convert to float and normalize
            content = content_np.astype(np.float32) / 255.0
            style = style_np.astype(np.float32) / 255.0
            
            # Ensure both images have the same size
            if content.shape != style.shape:
                style_resized = ndimage.zoom(style, (content.shape[0]/style.shape[0], 
                                                    content.shape[1]/style.shape[1], 1), order=1)
                style = style_resized
            
            # Convert to tensors
            content_tensor = tf.convert_to_tensor(content[np.newaxis, ...])
            style_tensor = tf.convert_to_tensor(style[np.newaxis, ...])
            
            # Extract features using VGG encoder
            content_features = self.vgg(content_tensor)
            style_features = self.vgg(style_tensor)
            
            # Apply WCT for style transfer
            stylized_features = self.whitening_and_coloring_transform(content_features, style_features)
            
            # Decode back to image
            stylized_tensor = self.decoder(stylized_features)
            stylized = stylized_tensor.numpy()[0]
            
            # Clip to valid range
            stylized = np.clip(stylized, 0, 1)
            
            # Blend with original content based on alpha
            blended = alpha * stylized + (1 - alpha) * content
            
            return (blended * 255).clip(0, 255).astype(np.uint8)
            
        except Exception as e:
            print(f"AST stylization error: {e}")
            import traceback
            traceback.print_exc()
            # Return simple blended content if stylization fails
            blended = alpha * style_np + (1 - alpha) * content_np
            return (blended * 255).clip(0, 255).astype(np.uint8)
    

    @staticmethod
    def metadata():
        return {
            "name": "AST (Arbitrary Style Transfer)",
            "type": "VGG19 + WCT + Decoder",
            "features": "Whitening & Coloring Transform, Covariance matching, Arbitrary styles",
            "paper": "Li et al. (2017) - Universal Style Transfer via Feature Transform",
            "real_time": True
        }
