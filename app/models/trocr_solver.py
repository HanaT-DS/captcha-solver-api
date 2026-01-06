"""
TrOCR Solver - Modèle TrOCR pré-entraîné pour les CAPTCHAs.

Utilise le modèle `DunnBC22/trocr-base-printed_captcha_ocr` de HuggingFace.

Performance:
- CER : 0.75% (Character Error Rate)
- Accuracy : ~99.25%
- Charset : Full alphanumeric (a-z, A-Z, 0-9)

Architecture:
- Encoder : Vision Transformer (ViT)
- Decoder : Transformer autorégressif
- Prétrainement : microsoft/trocr-base-printed

M2 MoSEF - Université Paris 1 Panthéon-Sorbonne
"""

import io
import time
from typing import Optional, Dict, Any

import numpy as np
from PIL import Image

from app.models.base_solver import BaseSolver, SolverResult, ModelType


class TrOCRSolver(BaseSolver):
    """
    Solver utilisant TrOCR fine-tuné pour les CAPTCHAs.
    
    Ce modèle est le plus précis pour la résolution de CAPTCHAs
    avec un charset complet (lettres + chiffres).
    
    Attributes:
        model_name: Nom du modèle HuggingFace.
        cache_dir: Répertoire de cache pour les poids.
    """
    
    # Modèle par défaut (99.25% accuracy)
    DEFAULT_MODEL = "DunnBC22/trocr-base-printed_captcha_ocr"
    
    # Charset complet supporté
    CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialise le solver TrOCR.
        
        Args:
            model_name: Nom du modèle HuggingFace (optionnel).
            cache_dir: Répertoire de cache (optionnel).
            device: Device PyTorch ('cpu' ou 'cuda').
        """
        super().__init__(
            name="trocr",
            model_type=ModelType.TROCR,
            charset=self.CHARSET,
        )
        
        self.model_name = model_name or self.DEFAULT_MODEL
        self.cache_dir = cache_dir
        self._requested_device = device
        
        # Modèle et processeur (chargés à la demande)
        self._model = None
        self._processor = None
        self._device = None
    
    @property
    def is_available(self) -> bool:
        """
        Vérifie si TrOCR peut être utilisé.
        
        Returns:
            True si transformers est installé.
        """
        try:
            import transformers
            return True
        except ImportError:
            return False
    
    def load(self) -> None:
        """
        Charge le modèle TrOCR depuis HuggingFace.
        
        Le modèle est téléchargé automatiquement au premier chargement
        et mis en cache pour les utilisations suivantes.
        
        Raises:
            RuntimeError: Si le chargement échoue.
        """
        if self._is_loaded:
            return
        
        import warnings
        import torch
        from transformers import AutoProcessor, VisionEncoderDecoderModel
        
        # Ignorer les warnings non critiques
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        # Détecter le device
        if self._requested_device:
            self._device = torch.device(self._requested_device)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Charger le processeur (sans use_fast pour compatibilité)
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )
        
        # Charger le modèle
        self._model = VisionEncoderDecoderModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )
        
        # Vérifier que le modèle est bien chargé
        if self._model is None:
            raise RuntimeError(f"Échec du chargement du modèle {self.model_name}")
        
        # Déplacer vers le device
        self._model.to(self._device)
        self._model.eval()
        
        self._is_loaded = True
    
    def solve(self, image_bytes: bytes) -> SolverResult:
        """
        Résout un CAPTCHA avec TrOCR.
        
        Args:
            image_bytes: Image au format PNG/JPG en bytes.
            
        Returns:
            SolverResult avec le texte prédit et la confiance.
        """
        start_time = time.time()
        
        # Charger le modèle si nécessaire
        self.ensure_loaded()
        
        import torch
        
        # Charger l'image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Prétraitement avec le processeur TrOCR
        pixel_values = self._processor(
            image,
            return_tensors="pt",
        ).pixel_values.to(self._device)
        
        # Génération avec calcul de la confiance
        with torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                max_length=32,
                num_beams=4,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        # Décoder le texte
        generated_ids = outputs.sequences
        text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        
        # Calculer la confiance
        confidence = self._compute_confidence(outputs)
        
        processing_time = (time.time() - start_time) * 1000
        
        return SolverResult(
            text=text,
            confidence=confidence,
            model=self.name,
            processing_time_ms=round(processing_time, 2),
            metadata={
                "model_name": self.model_name,
                "device": str(self._device),
                "num_beams": 4,
            },
        )
    
    def _compute_confidence(self, outputs) -> Optional[float]:
        """
        Calcule la confiance à partir des scores de génération.
        
        Args:
            outputs: Sortie de model.generate() avec output_scores=True.
            
        Returns:
            Score de confiance moyen (0-1).
        """
        import torch
        
        if not hasattr(outputs, "scores") or outputs.scores is None:
            return None
        
        try:
            # Récupérer les tokens générés (sans le token de départ)
            generated_ids = outputs.sequences[0, 1:]
            
            # Calculer la confiance pour chaque token
            confidences = []
            
            for step_idx, (score, token_id) in enumerate(zip(outputs.scores, generated_ids)):
                if token_id == self._processor.tokenizer.eos_token_id:
                    break
                
                # Softmax pour obtenir les probabilités
                probs = torch.softmax(score[0], dim=-1)
                token_prob = probs[token_id].item()
                confidences.append(token_prob)
            
            if confidences:
                # Moyenne géométrique pour éviter les produits très petits
                avg_confidence = np.exp(np.mean(np.log(confidences)))
                return round(float(avg_confidence), 3)
            
            return None
            
        except Exception:
            return None
    
    def unload(self) -> None:
        """Libère la mémoire du modèle."""
        import torch
        
        if self._model is not None:
            del self._model
            self._model = None
        
        if self._processor is not None:
            del self._processor
            self._processor = None
        
        # Libérer la mémoire GPU si disponible
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._is_loaded = False


class TrOCRSolverAlternative(TrOCRSolver):
    """
    Version alternative avec un modèle TrOCR différent.
    
    Utilise zenoda/trocr-captcha-killer (93.7% accuracy).
    Supporte l'anglais et le chinois.
    """
    
    DEFAULT_MODEL = "zenoda/trocr-captcha-killer"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._name = "trocr_alt"
        if "model_name" not in kwargs:
            self.model_name = self.DEFAULT_MODEL