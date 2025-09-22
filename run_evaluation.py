import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from models.registry import registry
from models.evaluation import StyleTransferEvaluator
import json

def run_comparative_evaluation():
    np.random.seed(42)
    
    test_results = {}
    models = ["adain", "adain_vgg", "wct", "microast"]
    
    evaluator = StyleTransferEvaluator()
    
    for model_name in models:
        print(f"Evaluating {model_name}...")
        
        model_metrics = []
        processing_times = []
        
        for i in range(10):
            content = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            style = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            
            result = registry.stylize_with_evaluation(model_name, content, style, alpha=1.0)
            
            if result['success']:
                model_metrics.append(result['evaluation_metrics'])
                processing_times.append(result['processing_time'])
        
        if model_metrics:
            avg_fsf = np.mean([m['fsf_score'] for m in model_metrics])
            avg_content = np.mean([m['content_preservation'] for m in model_metrics])
            avg_style = np.mean([m['style_alignment'] for m in model_metrics])
            avg_time = np.mean(processing_times)
            
            test_results[model_name] = {
                'fsf_score': round(avg_fsf, 3),
                'content_preservation': round(avg_content, 3),
                'style_alignment': round(avg_style, 3),
                'avg_processing_time': round(avg_time, 2)
            }
        else:
            test_results[model_name] = {
                'fsf_score': 0.0,
                'content_preservation': 0.0,
                'style_alignment': 0.0,
                'avg_processing_time': 0.0
            }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS FOR REPORT")
    print("="*80)
    print(f"{'Model':<12} {'FSF Score':<10} {'Content':<10} {'Style':<10} {'Time(s)':<10}")
    print("-"*60)
    
    for model, metrics in test_results.items():
        print(f"{model:<12} {metrics['fsf_score']:<10} {metrics['content_preservation']:<10} "
              f"{metrics['style_alignment']:<10} {metrics['avg_processing_time']:<10}")
    
    return test_results

if __name__ == "__main__":
    run_comparative_evaluation()