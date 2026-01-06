"""
CAPTCHA Factory API
===================
API de résolution automatique de CAPTCHAs visuels.

Modèles disponibles:
- CRNN : Modèle entraîné par Hana (98% accuracy)
- TrOCR : Modèle pré-entraîné HuggingFace (99% accuracy)
- Florence-2 : VLM zero-shot (universel)
- EasyOCR : OCR généraliste (fallback)

M2 MoSEF - Université Paris 1 Panthéon-Sorbonne
"""

__version__ = "2.0.0"
__author__ = "Hana & Aymen"
__project__ = "M2 MoSEF - Paris 1 Sorbonne"