from abc import ABC, abstractmethod
import time
import traceback
from typing import Dict, Any, Optional
import numpy as np
from .evaluation import StyleTransferEvaluator

class StyleTransferModel(ABC):
    
    @abstractmethod
    def stylize(self, content_np: np.ndarray, style_np: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        pass
    
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        pass
    
    def validate_inputs(self, content_np: np.ndarray, style_np: np.ndarray) -> bool:
        if content_np is None or style_np is None:
            raise ValueError("Input images cannot be None")
        
        if len(content_np.shape) != 3 or len(style_np.shape) != 3:
            raise ValueError("Images must be 3-dimensional (H, W, C)")
        
        if content_np.shape[2] != 3 or style_np.shape[2] != 3:
            raise ValueError("Images must have 3 channels (RGB)")
        
        return True

class ModelRegistry:
    
    def __init__(self):
        self._models = {}
        self._model_cache = {}
        self._performance_stats = {}
        self._error_counts = {}
        self._evaluator = StyleTransferEvaluator()
    
    def register_model(self, name: str, model_class, description: str = ""):
        self._models[name] = {
            'class': model_class,
            'description': description,
            'registered_at': time.time()
        }
        self._performance_stats[name] = {
            'total_calls': 0,
            'total_time': 0,
            'avg_time': 0,
            'success_rate': 0,
            'last_used': None
        }
        self._error_counts[name] = 0
    
    def get_model(self, name: str) -> StyleTransferModel:
        name = name.lower()
        
        if name not in self._models:
            available = list(self._models.keys())
            raise ValueError(f"Model '{name}' not found. Available models: {available}")
        
        if name in self._model_cache:
            return self._model_cache[name]
        
        try:
            model_class = self._models[name]['class']
            model_instance = model_class()
            
            self._model_cache[name] = model_instance
            
            return model_instance
            
        except Exception as e:
            self._error_counts[name] += 1
            print(f"Error creating model '{name}': {e}")
            raise
    
    def list_models(self) -> Dict[str, Any]:
        models_info = {}
        for name, info in self._models.items():
            models_info[name] = {
                'description': info['description'],
                'registered_at': info['registered_at'],
                'performance_stats': self._performance_stats[name],
                'error_count': self._error_counts[name],
                'cached': name in self._model_cache
            }
        return models_info
    
    def get_performance_stats(self) -> Dict[str, Any]:
        return {
            'models': self._performance_stats,
            'total_models': len(self._models),
            'cached_models': len(self._model_cache),
            'total_errors': sum(self._error_counts.values())
        }
    
    def stylize_with_tracking(self, model_name: str, content_np: np.ndarray, 
                            style_np: np.ndarray, alpha: float = 1.0) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            model = self.get_model(model_name)
            
            model.validate_inputs(content_np, style_np)
            
            result = model.stylize(content_np, style_np, alpha)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            self._update_performance_stats(model_name, processing_time, success=True)
            
            return {
                'success': True,
                'result': result,
                'processing_time': processing_time,
                'model_metadata': model.metadata(),
                'error': None
            }
            
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            
            self._update_performance_stats(model_name, processing_time, success=False)
            
            return {
                'success': False,
                'result': None,
                'processing_time': processing_time,
                'model_metadata': None,
                'error': str(e)
            }
    
    def stylize_with_evaluation(self, model_name: str, content_np: np.ndarray, 
                               style_np: np.ndarray, alpha: float = 1.0) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            model = self.get_model(model_name)
            model.validate_inputs(content_np, style_np)
            
            result = model.stylize(content_np, style_np, alpha)
            
            metrics = self._evaluator.compute_feature_statistical_fidelity(
                content_np, style_np, result
            )
            
            processing_time = time.time() - start_time
            self._update_performance_stats(model_name, processing_time, success=True)
            
            return {
                'success': True,
                'result': result,
                'processing_time': processing_time,
                'model_metadata': model.metadata(),
                'evaluation_metrics': metrics,
                'error': None
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_performance_stats(model_name, processing_time, success=False)
            
            return {
                'success': False,
                'result': None,
                'processing_time': processing_time,
                'model_metadata': None,
                'evaluation_metrics': None,
                'error': str(e)
            }
    
    def _update_performance_stats(self, model_name: str, processing_time: float, success: bool):
        stats = self._performance_stats[model_name]
        
        stats['total_calls'] += 1
        stats['total_time'] += processing_time
        stats['avg_time'] = stats['total_time'] / stats['total_calls']
        stats['last_used'] = time.time()
        
        if stats['total_calls'] == 1:
            stats['success_rate'] = 1.0 if success else 0.0
        else:
            alpha = 0.1
            stats['success_rate'] = alpha * (1.0 if success else 0.0) + (1 - alpha) * stats['success_rate']
    
    def clear_cache(self):
        self._model_cache.clear()

from app.models.adain import AdaINModel
from app.models.adain_vgg import AdaINVGGModel
from app.models.wct import WCTModel
from app.models.microast import MicroASTModel

registry = ModelRegistry()

registry.register_model("adain", AdaINModel, "Improved AdaIN with VGG19 encoder and reflection padding")
registry.register_model("adain_vgg", AdaINVGGModel, "AdaIN with VGG19 features and custom decoder")
registry.register_model("wct", WCTModel, "Improved WCT with multi-level processing and enhanced stability")
registry.register_model("microast", MicroASTModel, "Ultra-fast and ultra-resolution arbitrary style transfer with micro architecture")

def get_model(name: str):
    return registry.get_model(name)