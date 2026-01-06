"""
Tests pour le générateur de CAPTCHAs.

M2 MoSEF - Université Paris 1 Panthéon-Sorbonne
"""

import io
import pytest
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.captcha_generator import CaptchaGenerator, CharsetType, CHARSETS


class TestCaptchaGenerator:
    """Tests pour CaptchaGenerator."""
    
    def setup_method(self):
        """Setup avant chaque test."""
        self.generator = CaptchaGenerator()
    
    def test_generate_default(self):
        """Test génération avec paramètres par défaut."""
        image, text = self.generator.generate()
        
        assert isinstance(image, Image.Image)
        assert isinstance(text, str)
        assert len(text) == 5  # longueur par défaut
        assert image.size == (200, 60)  # dimensions par défaut
    
    def test_generate_custom_length(self):
        """Test génération avec longueur personnalisée."""
        for length in [3, 5, 8]:
            image, text = self.generator.generate(length=length)
            assert len(text) == length
    
    def test_generate_custom_dimensions(self):
        """Test génération avec dimensions personnalisées."""
        image, text = self.generator.generate(width=300, height=100)
        assert image.size == (300, 100)
    
    def test_generate_noise_levels(self):
        """Test génération avec différents niveaux de bruit."""
        for noise in [0.0, 0.3, 0.5, 1.0]:
            image, text = self.generator.generate(noise_level=noise)
            assert isinstance(image, Image.Image)
    
    def test_generate_custom_charset(self):
        """Test génération avec charset personnalisé."""
        charset = "ABC123"
        gen = CaptchaGenerator(charset=charset)
        image, text = gen.generate()
        
        assert all(c in charset for c in text)
    
    def test_generate_batch(self):
        """Test génération en batch."""
        results = self.generator.generate_batch(n=5)
        
        assert len(results) == 5
        for image, text in results:
            assert isinstance(image, Image.Image)
            assert isinstance(text, str)
    
    def test_generate_batch_variable_length(self):
        """Test génération batch avec longueur variable."""
        results = self.generator.generate_batch(n=10, length=8, variable_length=True)
        
        lengths = [len(text) for _, text in results]
        assert min(lengths) >= 3
        assert max(lengths) <= 8
    
    def test_charset_crnn(self):
        """Test charset CRNN (19 caractères)."""
        charset = CHARSETS[CharsetType.CRNN]
        assert len(charset) == 19
        assert "23456789" in charset  # chiffres sans 0 et 1
        assert "a" not in charset  # pas de 'a'
    
    def test_charset_info(self):
        """Test informations sur le charset."""
        info = self.generator.charset_info
        
        assert "charset" in info
        assert "size" in info
        assert info["size"] == len(self.generator.charset)
    
    def test_generate_for_model_crnn(self):
        """Test génération optimisée pour CRNN."""
        results = self.generator.generate_for_model(model="crnn", n=3)
        
        assert len(results) == 3
        for image, text in results:
            assert image.size == (300, 75)  # dimensions CRNN
    
    def test_image_is_rgb(self):
        """Test que l'image est en RGB."""
        image, _ = self.generator.generate()
        assert image.mode == "RGB"
    
    def test_image_saveable(self):
        """Test que l'image peut être sauvegardée."""
        image, _ = self.generator.generate()
        
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        
        assert len(buffer.getvalue()) > 0
    
    def test_text_in_charset(self):
        """Test que le texte généré est dans le charset."""
        charset = self.generator.charset
        
        for _ in range(10):
            _, text = self.generator.generate()
            assert all(c in charset for c in text)


class TestGeneratorLimits:
    """Tests des limites du générateur."""
    
    def setup_method(self):
        self.generator = CaptchaGenerator()
    
    def test_length_min(self):
        """Test longueur minimale."""
        _, text = self.generator.generate(length=1)
        assert len(text) >= 1
    
    def test_length_max(self):
        """Test longueur maximale."""
        _, text = self.generator.generate(length=10)
        assert len(text) <= 10
    
    def test_noise_clamped(self):
        """Test que le bruit est limité à [0, 1]."""
        # Ces valeurs devraient être clampées sans erreur
        self.generator.generate(noise_level=-0.5)
        self.generator.generate(noise_level=1.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])