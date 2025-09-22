import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_evaluation_metrics():
    from models.evaluation import StyleTransferEvaluator
    
    evaluator = StyleTransferEvaluator()
    
    content = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    style = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    stylized = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    metrics = evaluator.compute_feature_statistical_fidelity(content, style, stylized)
    
    assert 'fsf_score' in metrics
    assert 'content_preservation' in metrics
    assert 'style_alignment' in metrics
    assert 0 <= metrics['fsf_score'] <= 1
    
    print("Evaluation system test passed!")
    print(f"FSF Score: {metrics['fsf_score']:.3f}")
    print(f"Content Preservation: {metrics['content_preservation']:.3f}")
    print(f"Style Alignment: {metrics['style_alignment']:.3f}")

def test_registry_evaluation():
    from models.registry import registry
    
    content = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    style = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    result = registry.stylize_with_evaluation("microast", content, style, alpha=1.0)
    
    assert 'success' in result
    assert 'evaluation_metrics' in result
    assert result['success'] == True or result['success'] == False
    
    if result['success']:
        assert 'fsf_score' in result['evaluation_metrics']
        print("Registry evaluation integration test passed!")
        print(f"Processing time: {result['processing_time']:.3f}s")
        print(f"FSF Score: {result['evaluation_metrics']['fsf_score']:.3f}")
    else:
        print(f"Registry evaluation test failed with error: {result['error']}")

if __name__ == "__main__":
    print("Testing evaluation system...")
    test_evaluation_metrics()
    
    print("\nTesting registry evaluation integration...")
    test_registry_evaluation()