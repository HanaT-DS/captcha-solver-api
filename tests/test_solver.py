"""
Tests pour les solvers de CAPTCHAs.

M2 MoSEF - Université Paris 1 Panthéon-Sorbonne
"""

import io
import pytest
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.base_solver import BaseSolver, SolverResult, ModelType
from app.services.captcha_generator import CaptchaGenerator


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def generator():
    """Générateur de CAPTCHAs."""
    return CaptchaGenerator()


@pytest.fixture
def sample_image(generator):
    """Image CAPTCHA de test."""
    image, text = generator.generate(length=5, noise_level=0.2)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), text


# =============================================================================
# TESTS SOLVER RESULT
# =============================================================================

class TestSolverResult:
    """Tests pour SolverResult."""
    
    def test_create_result(self):
        """Test création d'un résultat."""
        result = SolverResult(
            text="abc123",
            confidence=0.95,
            model="test",
            processing_time_ms=100.0,
        )
        
        assert result.text == "abc123"
        assert result.confidence == 0.95
        assert result.model == "test"
        assert result.processing_time_ms == 100.0
    
    def test_to_dict(self):
        """Test conversion en dictionnaire."""
        result = SolverResult(
            text="test",
            confidence=0.8,
            model="trocr",
            processing_time_ms=50.0,
        )
        
        d = result.to_dict()
        
        assert d["text"] == "test"
        assert d["confidence"] == 0.8
        assert d["model"] == "trocr"
    
    def test_is_confident_true(self):
        """Test is_confident avec confiance élevée."""
        result = SolverResult(
            text="test",
            confidence=0.9,
            model="test",
            processing_time_ms=10.0,
        )
        assert result.is_confident is True
    
    def test_is_confident_false(self):
        """Test is_confident avec confiance basse."""
        result = SolverResult(
            text="test",
            confidence=0.5,
            model="test",
            processing_time_ms=10.0,
        )
        assert result.is_confident is False
    
    def test_is_confident_none(self):
        """Test is_confident sans confiance."""
        result = SolverResult(
            text="test",
            confidence=None,
            model="test",
            processing_time_ms=10.0,
        )
        assert result.is_confident is False


# =============================================================================
# TESTS SOLVER SERVICE
# =============================================================================

class TestSolverService:
    """Tests pour SolverService."""
    
    @pytest.fixture
    def solver_service(self):
        """Service de résolution."""
        from app.services.solver_service import SolverService
        return SolverService()
    
    def test_available_models(self, solver_service):
        """Test liste des modèles disponibles."""
        models = solver_service.available_models
        assert isinstance(models, list)
        # Au moins un modèle devrait être disponible
        assert len(models) > 0
    
    def test_get_model_info(self, solver_service):
        """Test informations sur les modèles."""
        info = solver_service.get_model_info()
        assert isinstance(info, dict)
        
        for name, model_info in info.items():
            assert "name" in model_info
            assert "type" in model_info
            assert "is_loaded" in model_info
    
    @pytest.mark.slow
    def test_solve_trocr(self, solver_service, sample_image):
        """Test résolution avec TrOCR."""
        image_bytes, true_text = sample_image
        
        if "trocr" not in solver_service.available_models:
            pytest.skip("TrOCR non disponible")
        
        result = solver_service.solve(image_bytes, model="trocr")
        
        assert isinstance(result, SolverResult)
        assert result.text  # Non vide
        assert result.model == "trocr"
        assert result.processing_time_ms > 0
    
    def test_solve_invalid_model(self, solver_service, sample_image):
        """Test résolution avec modèle invalide."""
        image_bytes, _ = sample_image
        
        with pytest.raises(ValueError):
            solver_service.solve(image_bytes, model="invalid_model")
    
    @pytest.mark.slow
    def test_compare(self, solver_service, sample_image):
        """Test comparaison des modèles."""
        image_bytes, true_text = sample_image
        
        result = solver_service.compare(image_bytes, true_text=true_text)
        
        assert hasattr(result, "results")
        assert hasattr(result, "winner")
        assert isinstance(result.results, dict)


# =============================================================================
# TESTS CRNN SOLVER
# =============================================================================

class TestCRNNSolver:
    """Tests pour le solver CRNN."""
    
    def test_charset(self):
        """Test charset CRNN."""
        from app.models.crnn_model import CRNNSolver
        
        solver = CRNNSolver()
        assert len(solver.charset) == 19
        assert solver.charset == "23456789bcdefgmnpwxy"
    
    def test_supports_text(self):
        """Test vérification de support de texte."""
        from app.models.crnn_model import CRNNSolver
        
        solver = CRNNSolver()
        
        assert solver.supports_text("b3c4d")  # Valide
        assert not solver.supports_text("abc")  # 'a' non supporté
        assert not solver.supports_text("ABC")  # Majuscules non supportées


# =============================================================================
# TESTS TROCR SOLVER
# =============================================================================

class TestTrOCRSolver:
    """Tests pour le solver TrOCR."""
    
    def test_charset(self):
        """Test charset TrOCR."""
        from app.models.trocr_solver import TrOCRSolver
        
        solver = TrOCRSolver()
        assert len(solver.charset) == 62  # a-z + A-Z + 0-9
    
    def test_is_available(self):
        """Test disponibilité TrOCR."""
        from app.models.trocr_solver import TrOCRSolver
        
        solver = TrOCRSolver()
        # Devrait toujours être True si transformers est installé
        assert isinstance(solver.is_available, bool)


# =============================================================================
# TESTS UTILS
# =============================================================================

class TestCharsetUtils:
    """Tests pour les utilitaires de charset."""
    
    def test_compute_cer(self):
        """Test calcul du CER."""
        from app.utils.charset import compute_cer
        
        assert compute_cer("abc", "abc") == 0.0
        assert compute_cer("abc", "abd") > 0
        assert compute_cer("", "abc") == 1.0
    
    def test_compute_accuracy(self):
        """Test calcul de l'accuracy."""
        from app.utils.charset import compute_accuracy
        
        assert compute_accuracy("abc", "abc") == 1.0
        assert compute_accuracy("abc", "abd") == 0.0
        assert compute_accuracy("ABC", "abc", case_sensitive=False) == 1.0
    
    def test_normalize_text(self):
        """Test normalisation de texte."""
        from app.utils.charset import normalize_text
        
        assert normalize_text("AbC123") == "abc123"
        assert normalize_text("a b c") == "abc"
        assert normalize_text("a@b#c") == "abc"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])