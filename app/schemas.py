"""
Schémas Pydantic pour la validation des requêtes et réponses de l'API.
"""
from pydantic import BaseModel, Field
from typing import Optional


class Health(BaseModel):
    """Schéma de réponse pour l'état du système."""
    status: str = Field(..., example="ok")
    version: str = Field("1.0.0", example="1.0.0")


class CaptchaResponse(BaseModel):
    """Schéma de réponse renvoyé après la résolution d'une image."""
    text: str = Field(..., description="Le texte extrait du CAPTCHA", example="2n5bb")
    filename: Optional[str] = Field(None, description="Le nom du fichier original")
    confidence: float = Field(0.0, description="Score de confiance du modèle (0 à 1)")