"""
Solver module - EasyOCR pour la resolution de CAPTCHAs.

Ce module fournit une classe simple pour resoudre les CAPTCHAs
en utilisant EasyOCR comme moteur OCR.
"""

import io
import logging
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)


class CaptchaSolver:
    """
    Classe pour resoudre les CAPTCHAs avec EasyOCR.
    
    Attributes:
        languages: Liste des langues supportees par l'OCR.
    """
    
    def __init__(self, languages: Optional[List[str]] = None) -> None:
        """
        Initialise le solver EasyOCR.
        
        Args:
            languages: Liste des langues pour l'OCR. Par defaut ['en'].
        """
        self._languages = languages or ["en"]
        self._reader = None
        logger.info(f"CaptchaSolver initialise avec langues: {self._languages}")
    
    def _ensure_loaded(self) -> None:
        """
        Charge le modele EasyOCR si necessaire.
        
        Le chargement est fait de maniere paresseuse (lazy loading)
        pour eviter de charger le modele au demarrage de l'API.
        """
        if self._reader is None:
            import easyocr
            logger.info("Chargement du modele EasyOCR...")
            self._reader = easyocr.Reader(self._languages, gpu=False)
            logger.info("Modele EasyOCR charge")
    
    def solve(self, image_bytes: bytes) -> Tuple[str, float]:
        """
        Resout un CAPTCHA a partir des bytes de l'image.
        
        Args:
            image_bytes: Image au format bytes (PNG/JPG).
            
        Returns:
            Tuple contenant:
                - text: Le texte detecte dans l'image.
                - confidence: Score de confiance entre 0 et 1.
                
        Raises:
            ValueError: Si l'image ne peut pas etre lue.
        """
        self._ensure_loaded()
        
        # Convertir bytes en image PIL puis en numpy array
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_array = np.array(image)
        except Exception as e:
            logger.error(f"Erreur lors de la lecture de l'image: {e}")
            raise ValueError(f"Image invalide: {e}")
        
        # Lire le texte avec EasyOCR
        results = self._reader.readtext(image_array)
        
        if not results:
            logger.warning("Aucun texte detecte dans l'image")
            return "", 0.0
        
        # Extraire les textes et confidences
        texts = []
        confidences = []
        
        for bbox, text, confidence in results:
            texts.append(text)
            confidences.append(confidence)
            logger.debug(f"Detecte: '{text}' (confiance: {confidence:.3f})")
        
        # Concatener tous les textes
        full_text = "".join(texts)
        
        # Calculer la confiance moyenne
        avg_confidence = sum(confidences) / len(confidences)
        
        # Nettoyer le texte (garder uniquement les caracteres alphanumeriques)
        clean_text = "".join(char for char in full_text if char.isalnum())
        
        logger.info(f"Resultat: '{clean_text}' (confiance: {avg_confidence:.3f})")
        
        return clean_text, round(avg_confidence, 3)
