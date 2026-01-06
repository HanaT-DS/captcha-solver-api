"""
Tests pour l'API FastAPI.

M2 MoSEF - Université Paris 1 Panthéon-Sorbonne
"""

import base64
import io
import pytest
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def client():
    """Client de test FastAPI."""
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


@pytest.fixture
def sample_captcha_base64():
    """CAPTCHA en base64 pour les tests."""
    from app.services.captcha_generator import CaptchaGenerator
    
    generator = CaptchaGenerator()
    image, text = generator.generate(length=5, noise_level=0.2)
    
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return image_base64, text


# =============================================================================
# TESTS ROOT & HEALTH
# =============================================================================

class TestRootEndpoints:
    """Tests pour les endpoints root et health."""
    
    def test_root(self, client):
        """Test endpoint root."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "models" in data
    
    def test_health(self, client):
        """Test endpoint health."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models_available" in data
    
    def test_models(self, client):
        """Test endpoint models."""
        response = client.get("/models")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


# =============================================================================
# TESTS GÉNÉRATION
# =============================================================================

class TestGenerateEndpoints:
    """Tests pour les endpoints de génération."""
    
    def test_generate_default(self, client):
        """Test génération par défaut."""
        response = client.post("/generate", json={})
        
        assert response.status_code == 200
        data = response.json()
        assert "image_base64" in data
        assert "true_text" in data
        assert len(data["true_text"]) == 5
    
    def test_generate_custom(self, client):
        """Test génération avec paramètres."""
        response = client.post("/generate", json={
            "length": 8,
            "width": 300,
            "height": 100,
            "noise_level": 0.5,
        })
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["true_text"]) == 8
        assert data["width"] == 300
        assert data["height"] == 100
    
    def test_generate_random(self, client):
        """Test génération aléatoire."""
        response = client.get("/generate/random")
        
        assert response.status_code == 200
        data = response.json()
        assert "image_base64" in data
        assert "true_text" in data


# =============================================================================
# TESTS RÉSOLUTION
# =============================================================================

class TestSolveEndpoints:
    """Tests pour les endpoints de résolution."""
    
    @pytest.mark.slow
    def test_solve_base64(self, client, sample_captcha_base64):
        """Test résolution base64."""
        image_base64, true_text = sample_captcha_base64
        
        response = client.post("/solve", json={
            "image_base64": image_base64,
            "model": "trocr",
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "predicted_text" in data
        assert "model_used" in data
        assert "processing_time_ms" in data
    
    def test_solve_invalid_base64(self, client):
        """Test résolution avec base64 invalide."""
        response = client.post("/solve", json={
            "image_base64": "invalid_base64",
            "model": "trocr",
        })
        
        assert response.status_code == 400
    
    def test_solve_invalid_model(self, client, sample_captcha_base64):
        """Test résolution avec modèle invalide."""
        image_base64, _ = sample_captcha_base64
        
        response = client.post("/solve", json={
            "image_base64": image_base64,
            "model": "invalid_model",
        })
        
        assert response.status_code == 400
    
    @pytest.mark.slow
    def test_solve_cascade(self, client, sample_captcha_base64):
        """Test résolution en cascade."""
        image_base64, _ = sample_captcha_base64
        
        response = client.post("/solve/cascade", json={
            "image_base64": image_base64,
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "predicted_text" in data


# =============================================================================
# TESTS COMPARAISON
# =============================================================================

class TestCompareEndpoints:
    """Tests pour les endpoints de comparaison."""
    
    @pytest.mark.slow
    def test_compare(self, client, sample_captcha_base64):
        """Test comparaison des modèles."""
        image_base64, true_text = sample_captcha_base64
        
        response = client.post("/compare", json={
            "image_base64": image_base64,
            "true_text": true_text,
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "winner" in data
    
    @pytest.mark.slow
    def test_benchmark(self, client):
        """Test benchmark."""
        response = client.get("/benchmark", params={
            "n_samples": 3,
            "noise_level": 0.3,
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "n_samples" in data
        assert "summary" in data


# =============================================================================
# TESTS UPLOAD
# =============================================================================

class TestUploadEndpoints:
    """Tests pour les endpoints d'upload."""
    
    def test_solve_upload_invalid_type(self, client):
        """Test upload avec type invalide."""
        response = client.post(
            "/solve/upload",
            files={"file": ("test.txt", b"not an image", "text/plain")},
            data={"model": "trocr"},
        )
        
        assert response.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])