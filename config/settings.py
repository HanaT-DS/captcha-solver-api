"""
Configuration globale du projet CAPTCHA Factory.

Ce module centralise tous les paramètres de configuration
pour faciliter la maintenance et le déploiement.

M2 MoSEF - Université Paris 1 Panthéon-Sorbonne
"""

import os
from pathlib import Path
from typing import Literal, Optional
from dataclasses import dataclass, field
from functools import lru_cache


# =============================================================================
# CHEMINS DU PROJET
# =============================================================================

# Racine du projet (parent de config/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
APP_DIR = PROJECT_ROOT / "app"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
CACHE_DIR = MODELS_DIR / "cache"

# Créer les dossiers s'ils n'existent pas
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION DES MODÈLES
# =============================================================================

@dataclass
class CRNNConfig:
    """Configuration du modèle CRNN (Hana)."""
    
    model_path: Path = MODELS_DIR / "captcha_model.pth"
    mapping_path: Path = MODELS_DIR / "captcha_model_mapping.joblib"
    
    # Dimensions d'entrée
    image_width: int = 300
    image_height: int = 75
    
    # Charset supporté (19 caractères)
    charset: str = "23456789bcdefgmnpwxy"
    
    # Architecture
    num_chars: int = 19
    hidden_size: int = 32
    num_layers: int = 2
    dropout: float = 0.25
    
    @property
    def is_available(self) -> bool:
        """Vérifie si les fichiers du modèle existent."""
        return self.model_path.exists() and self.mapping_path.exists()


@dataclass
class TrOCRConfig:
    """Configuration du modèle TrOCR pré-entraîné."""
    
    # Modèle HuggingFace (99.25% accuracy sur CAPTCHAs)
    model_name: str = "DunnBC22/trocr-base-printed_captcha_ocr"
    
    # Alternative: modèle plus léger
    # model_name: str = "microsoft/trocr-small-printed"
    
    # Cache local
    cache_dir: Path = CACHE_DIR / "trocr"
    
    # Paramètres d'inférence
    max_length: int = 32
    num_beams: int = 4
    
    # Charset supporté (complet)
    charset: str = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    @property
    def is_available(self) -> bool:
        """Vérifie si le modèle peut être chargé (toujours True car téléchargeable)."""
        return True


@dataclass
class FlorenceConfig:
    """Configuration du modèle Florence-2 (VLM)."""
    
    # Modèle HuggingFace (zero-shot OCR)
    model_name: str = "microsoft/Florence-2-base"
    
    # Alternative: modèle plus grand pour meilleure accuracy
    # model_name: str = "microsoft/Florence-2-large"
    
    # Cache local
    cache_dir: Path = CACHE_DIR / "florence"
    
    # Paramètres d'inférence
    max_new_tokens: int = 64
    num_beams: int = 3
    
    # Prompt pour OCR
    ocr_prompt: str = "<OCR>"
    
    @property
    def is_available(self) -> bool:
        """Vérifie si le modèle peut être chargé (toujours True car téléchargeable)."""
        return True


@dataclass
class EasyOCRConfig:
    """Configuration d'EasyOCR (fallback)."""
    
    languages: list = field(default_factory=lambda: ["en"])
    gpu: bool = False
    
    @property
    def is_available(self) -> bool:
        """Vérifie si EasyOCR est installé."""
        try:
            import easyocr
            return True
        except ImportError:
            return False


# =============================================================================
# CONFIGURATION DU GÉNÉRATEUR
# =============================================================================

@dataclass
class GeneratorConfig:
    """Configuration du générateur de CAPTCHAs."""
    
    # Charset par défaut (compatible CRNN)
    default_charset: str = "23456789bcdefgmnpwxy"
    
    # Charset étendu (pour tests)
    extended_charset: str = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Dimensions par défaut
    default_width: int = 200
    default_height: int = 60
    default_length: int = 5
    default_noise_level: float = 0.3
    
    # Limites
    min_length: int = 1
    max_length: int = 10
    min_width: int = 100
    max_width: int = 400
    min_height: int = 40
    max_height: int = 150


# =============================================================================
# CONFIGURATION DU SERVICE DE RÉSOLUTION
# =============================================================================

@dataclass
class SolverConfig:
    """Configuration du service de résolution multi-modèle."""
    
    # Modèle par défaut
    default_model: Literal["crnn", "trocr", "florence", "easyocr", "cascade"] = "cascade"
    
    # Seuil de confiance pour la cascade
    confidence_threshold: float = 0.85
    
    # Ordre de priorité pour la cascade
    cascade_order: list = field(default_factory=lambda: ["crnn", "trocr", "florence"])
    
    # Timeout par modèle (en secondes)
    model_timeout: float = 30.0
    
    # Préchargement des modèles au démarrage
    preload_models: list = field(default_factory=lambda: ["trocr"])


# =============================================================================
# CONFIGURATION DE L'API
# =============================================================================

@dataclass
class APIConfig:
    """Configuration de l'API FastAPI."""
    
    title: str = "CAPTCHA Factory API"
    description: str = "API de résolution automatique de CAPTCHAs visuels"
    version: str = "2.0.0"
    
    # CORS
    cors_origins: list = field(default_factory=lambda: ["*"])
    cors_methods: list = field(default_factory=lambda: ["*"])
    cors_headers: list = field(default_factory=lambda: ["*"])
    
    # Limites
    max_upload_size: int = 10 * 1024 * 1024  # 10 MB
    rate_limit: int = 100  # requêtes par minute


# =============================================================================
# CONFIGURATION DU DASHBOARD
# =============================================================================

@dataclass
class DashboardConfig:
    """Configuration du dashboard Streamlit."""
    
    page_title: str = "CAPTCHA Factory"
    page_icon: str = "🔓"
    layout: str = "wide"


# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

@dataclass
class Settings:
    """Configuration globale du projet."""
    
    # Device PyTorch
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "cpu"))
    
    # Niveau de log
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    # Configurations des composants
    crnn: CRNNConfig = field(default_factory=CRNNConfig)
    trocr: TrOCRConfig = field(default_factory=TrOCRConfig)
    florence: FlorenceConfig = field(default_factory=FlorenceConfig)
    easyocr: EasyOCRConfig = field(default_factory=EasyOCRConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    api: APIConfig = field(default_factory=APIConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    
    # Chemins
    project_root: Path = PROJECT_ROOT
    models_dir: Path = MODELS_DIR
    data_dir: Path = DATA_DIR
    cache_dir: Path = CACHE_DIR


@lru_cache()
def get_settings() -> Settings:
    """
    Récupère l'instance singleton des settings.
    
    Returns:
        Instance unique de Settings.
    """
    return Settings()


# =============================================================================
# CONSTANTES
# =============================================================================

# Charsets prédéfinis
CHARSET_CRNN = "23456789bcdefgmnpwxy"  # 19 caractères (dataset original)
CHARSET_DIGITS = "0123456789"  # 10 chiffres
CHARSET_LOWERCASE = "abcdefghijklmnopqrstuvwxyz"  # 26 lettres minuscules
CHARSET_UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 26 lettres majuscules
CHARSET_ALPHANUMERIC = CHARSET_DIGITS + CHARSET_LOWERCASE + CHARSET_UPPERCASE  # 62 caractères

# Normalisation ImageNet (utilisée par tous les modèles)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Types MIME supportés
SUPPORTED_IMAGE_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/webp"]

# Modèles disponibles
AVAILABLE_MODELS = ["crnn", "trocr", "florence", "easyocr", "cascade"]