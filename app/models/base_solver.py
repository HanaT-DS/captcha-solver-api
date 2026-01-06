"""
Base Solver - Classe abstraite pour tous les solveurs de CAPTCHA.

Définit l'interface commune que tous les modèles de résolution
doivent implémenter.

M2 MoSEF - Université Paris 1 Panthéon-Sorbonne
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum


class ModelType(Enum):
    """Types de modèles disponibles."""
    CRNN = "crnn"
    TROCR = "trocr"
    FLORENCE = "florence"
    EASYOCR = "easyocr"


@dataclass
class SolverResult:
    """
    Résultat d'une résolution de CAPTCHA.
    
    Attributes:
        text: Texte prédit.
        confidence: Score de confiance (0-1), None si non disponible.
        model: Nom du modèle utilisé.
        processing_time_ms: Temps de traitement en millisecondes.
        metadata: Métadonnées supplémentaires.
    """
    text: str
    confidence: Optional[float]
    model: str
    processing_time_ms: float
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit le résultat en dictionnaire."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "model": self.model,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata or {},
        }
    
    @property
    def is_confident(self) -> bool:
        """Vérifie si la prédiction est confiante (>= 0.85)."""
        if self.confidence is None:
            return False
        return self.confidence >= 0.85


class BaseSolver(ABC):
    """
    Classe abstraite pour tous les solveurs de CAPTCHA.
    
    Chaque solver doit implémenter:
    - load(): Charger le modèle en mémoire
    - solve(): Résoudre un CAPTCHA
    - unload(): Libérer la mémoire (optionnel)
    
    Attributes:
        name: Nom du solver.
        model_type: Type de modèle (CRNN, TrOCR, etc.).
        charset: Ensemble des caractères supportés.
        is_loaded: Indique si le modèle est chargé.
    """
    
    def __init__(
        self,
        name: str,
        model_type: ModelType,
        charset: str,
    ) -> None:
        """
        Initialise le solver.
        
        Args:
            name: Nom unique du solver.
            model_type: Type de modèle.
            charset: Caractères supportés par le modèle.
        """
        self._name = name
        self._model_type = model_type
        self._charset = charset
        self._is_loaded = False
    
    @property
    def name(self) -> str:
        """Nom du solver."""
        return self._name
    
    @property
    def model_type(self) -> ModelType:
        """Type de modèle."""
        return self._model_type
    
    @property
    def charset(self) -> str:
        """Caractères supportés."""
        return self._charset
    
    @property
    def charset_set(self) -> set:
        """Ensemble des caractères supportés (pour vérification rapide)."""
        return set(self._charset)
    
    @property
    def is_loaded(self) -> bool:
        """Indique si le modèle est chargé en mémoire."""
        return self._is_loaded
    
    def supports_text(self, text: str) -> bool:
        """
        Vérifie si le solver supporte tous les caractères du texte.
        
        Args:
            text: Texte à vérifier.
            
        Returns:
            True si tous les caractères sont dans le charset.
        """
        return all(c in self.charset_set for c in text.lower())
    
    @abstractmethod
    def load(self) -> None:
        """
        Charge le modèle en mémoire.
        
        Raises:
            RuntimeError: Si le chargement échoue.
        """
        pass
    
    @abstractmethod
    def solve(self, image_bytes: bytes) -> SolverResult:
        """
        Résout un CAPTCHA à partir des bytes de l'image.
        
        Args:
            image_bytes: Image au format PNG/JPG en bytes.
            
        Returns:
            SolverResult avec le texte prédit et les métadonnées.
            
        Raises:
            RuntimeError: Si la résolution échoue.
        """
        pass
    
    def unload(self) -> None:
        """
        Libère la mémoire du modèle.
        
        Implémentation par défaut : ne fait rien.
        Les sous-classes peuvent surcharger pour libérer les ressources.
        """
        self._is_loaded = False
    
    def ensure_loaded(self) -> None:
        """Charge le modèle s'il n'est pas déjà chargé."""
        if not self._is_loaded:
            self.load()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', loaded={self.is_loaded})"
    
    def __str__(self) -> str:
        status = "✓" if self.is_loaded else "✗"
        return f"[{status}] {self.name} ({self.model_type.value})"