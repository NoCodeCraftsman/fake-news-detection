import pytest
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from explainability.explainer import FakeNewsExplainer, find_best_model

class TestExplainability:
    
    @pytest.fixture
    def sample_texts(self):
        return [
            "Scientists at MIT have developed a new method for detecting fake news.",
            "BREAKING: Government secretly implants microchips through vaccines.",
            "Weather forecast predicts rain for tomorrow in New York City."
        ]
    
    def test_find_best_model(self):
        """Test finding best model"""
        model_path = find_best_model("bert")
        if model_path:
            assert model_path.exists()
            assert (model_path / "config.json").exists()
    
    def test_explainer_initialization(self):
        """Test explainer initialization"""
        model_path = find_best_model("bert")
        if model_path:
            explainer = FakeNewsExplainer(model_path, "bert")
            assert explainer.model is not None
            assert explainer.tokenizer is not None
    
    def test_lime_explanation(self, sample_texts):
        """Test LIME explanation"""
        model_path = find_best_model("bert")
        if model_path:
            explainer = FakeNewsExplainer(model_path, "bert")
            
            for text in sample_texts:
                explanation = explainer.explain_with_lime(text)
                assert "method" in explanation
                assert explanation["method"] == "LIME"
                if "error" not in explanation:
                    assert "prediction" in explanation
                    assert "confidence" in explanation
                    assert "feature_importance" in explanation
    
    def test_shap_explanation(self, sample_texts):
        """Test SHAP explanation"""
        model_path = find_best_model("bert")
        if model_path:
            explainer = FakeNewsExplainer(model_path, "bert")
            
            for text in sample_texts[:1]:  # Test with one sample (SHAP is slower)
                explanation = explainer.explain_with_shap(text)
                assert "method" in explanation
                assert explanation["method"] == "SHAP"
                if "error" not in explanation:
                    assert "prediction" in explanation
                    assert "confidence" in explanation
                    assert "token_importance" in explanation
    
    def test_comprehensive_explanation(self, sample_texts):
        """Test comprehensive explanation"""
        model_path = find_best_model("bert")
        if model_path:
            explainer = FakeNewsExplainer(model_path, "bert")
            
            text = sample_texts[0]
            results = explainer.explain(text, methods=["lime"])
            
            assert "text" in results
            assert "explanations" in results
            assert "lime" in results["explanations"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])