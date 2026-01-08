"""
API FastAPI pour la resolution de CAPTCHAs.

Ce module expose un endpoint POST /solve qui recoit une image
en Base64 et retourne le texte detecte.
"""

import base64
import logging
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from api.solver import CaptchaSolver


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# MODELES PYDANTIC
# =============================================================================

class SolveRequest(BaseModel):
    """
    Requete pour resoudre un CAPTCHA.
    
    Attributes:
        image_base64: Image encodee en Base64.
    """
    image_base64: str = Field(
        ...,
        description="Image du CAPTCHA encodee en Base64"
    )


class SolveResponse(BaseModel):
    """
    Reponse avec le texte resolu.
    
    Attributes:
        success: True si la resolution a reussi.
        text: Texte detecte dans l'image.
        confidence: Score de confiance (0-1).
        error: Message d'erreur si echec.
    """
    success: bool
    text: str
    confidence: float
    error: Optional[str] = None


# =============================================================================
# APPLICATION FASTAPI
# =============================================================================

app = FastAPI(
    title="CAPTCHA Solver API",
    description="API pour resoudre les CAPTCHAs avec EasyOCR",
    version="1.0.0",
)

# Instance du solver (singleton)
solver = CaptchaSolver()


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health")
def health_check() -> dict:
    """
    Verifie que l'API est en fonctionnement.
    
    Returns:
        Dictionnaire avec le statut de l'API.
    """
    return {"status": "ok", "message": "API is running"}


@app.post("/solve", response_model=SolveResponse)
def solve_captcha(request: SolveRequest) -> SolveResponse:
    """
    Resout un CAPTCHA a partir d'une image en Base64.
    
    Args:
        request: Requete contenant l'image en Base64.
        
    Returns:
        Reponse avec le texte predit et la confiance.
    """
    logger.info("Reception d'une requete de resolution")
    
    try:
        # Decoder le Base64
        try:
            image_bytes = base64.b64decode(request.image_base64)
        except Exception as e:
            logger.error(f"Erreur de decodage Base64: {e}")
            return SolveResponse(
                success=False,
                text="",
                confidence=0.0,
                error=f"Base64 invalide: {e}"
            )
        
        logger.info(f"Image recue: {len(image_bytes)} bytes")
        
        # Resoudre avec EasyOCR
        text, confidence = solver.solve(image_bytes)
        
        logger.info(f"Resolution reussie: '{text}' (confiance: {confidence})")
        
        return SolveResponse(
            success=True,
            text=text,
            confidence=confidence
        )
    
    except Exception as e:
        logger.error(f"Erreur lors de la resolution: {e}")
        return SolveResponse(
            success=False,
            text="",
            confidence=0.0,
            error=str(e)
        )
