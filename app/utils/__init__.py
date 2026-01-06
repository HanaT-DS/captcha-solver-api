"""
Utilities package for CAPTCHA Factory.

Contient des fonctions utilitaires pour:
- Prétraitement d'images
- Calcul de confiance
- Gestion des charsets

M2 MoSEF - Université Paris 1 Panthéon-Sorbonne
"""

from app.utils.image_processing import (
    load_image_from_bytes,
    preprocess_for_crnn,
    preprocess_for_trocr,
    image_to_base64,
    base64_to_image,
    resize_image,
    normalize_image,
)

from app.utils.confidence import (
    compute_ctc_confidence,
    compute_sequence_confidence,
    calibrate_confidence,
    confidence_threshold_check,
)

from app.utils.charset import (
    validate_charset,
    is_charset_compatible,
    normalize_text,
    compute_cer,
    compute_accuracy,
)

__all__ = [
    # Image processing
    "load_image_from_bytes",
    "preprocess_for_crnn",
    "preprocess_for_trocr",
    "image_to_base64",
    "base64_to_image",
    "resize_image",
    "normalize_image",
    # Confidence
    "compute_ctc_confidence",
    "compute_sequence_confidence",
    "calibrate_confidence",
    "confidence_threshold_check",
    # Charset
    "validate_charset",
    "is_charset_compatible",
    "normalize_text",
    "compute_cer",
    "compute_accuracy",
]