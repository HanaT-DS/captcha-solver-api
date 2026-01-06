"""
Configuration package for CAPTCHA Factory.
"""

from config.settings import (
    Settings,
    get_settings,
    CRNNConfig,
    TrOCRConfig,
    FlorenceConfig,
    EasyOCRConfig,
    GeneratorConfig,
    SolverConfig,
    APIConfig,
    DashboardConfig,
    # Constantes
    CHARSET_CRNN,
    CHARSET_DIGITS,
    CHARSET_LOWERCASE,
    CHARSET_UPPERCASE,
    CHARSET_ALPHANUMERIC,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SUPPORTED_IMAGE_TYPES,
    AVAILABLE_MODELS,
    # Chemins
    PROJECT_ROOT,
    MODELS_DIR,
    DATA_DIR,
    CACHE_DIR,
)

__all__ = [
    "Settings",
    "get_settings",
    "CRNNConfig",
    "TrOCRConfig",
    "FlorenceConfig",
    "EasyOCRConfig",
    "GeneratorConfig",
    "SolverConfig",
    "APIConfig",
    "DashboardConfig",
    "CHARSET_CRNN",
    "CHARSET_DIGITS",
    "CHARSET_LOWERCASE",
    "CHARSET_UPPERCASE",
    "CHARSET_ALPHANUMERIC",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "SUPPORTED_IMAGE_TYPES",
    "AVAILABLE_MODELS",
    "PROJECT_ROOT",
    "MODELS_DIR",
    "DATA_DIR",
    "CACHE_DIR",
]