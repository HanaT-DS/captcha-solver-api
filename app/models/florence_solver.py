"""
Florence-2 Solver - Vision Language Model pour OCR zero-shot.

Utilise le modèle `microsoft/Florence-2-base` de HuggingFace.

Performance:
- Zero-shot : ~70-85% accuracy sur CAPTCHAs simples
- Universel : Supporte n'importe quel charset
- Flexible : Fonctionne sur des images de toutes tailles

Architecture:
- Encoder : DaViT (Dual Attention Vision Transformer)
- Decoder : Transformer multimodal
- Prétrainement : FLD-5B (5.4B annotations, 126M images)

M2 MoSEF - Université Paris 1 Panthéon-Sorbonne
"""

import io
import re
import time
from typing import Optional, Dict, Any

from PIL import Image

from app.models.base_solver import BaseSolver, SolverResult, ModelType


class FlorenceSolver(BaseSolver):
    """
    Solver utilisant Florence-2 comme VLM pour OCR zero-shot.
    
    Ce modèle est utilisé comme fallback car il peut lire
    n'importe quel texte sans fine-tuning préalable.
    
    Attributes:
        model_name: Nom du modèle HuggingFace.
        ocr_prompt: Prompt utilisé pour l'OCR.
    """
    
    # Modèle par défaut
    DEFAULT_MODEL = "microsoft/Florence-2-base"
    
    # Prompt pour l'OCR
    OCR_PROMPT = "<OCR>"
    
    # Charset universel
    CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+-=[]{}|;':\",./<>?"
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialise le solver Florence-2.
        
        Args:
            model_name: Nom du modèle HuggingFace (optionnel).
            cache_dir: Répertoire de cache (optionnel).
            device: Device PyTorch ('cpu' ou 'cuda').
        """
        super().__init__(
            name="florence",
            model_type=ModelType.FLORENCE,
            charset=self.CHARSET,
        )
        
        self.model_name = model_name or self.DEFAULT_MODEL
        self.cache_dir = cache_dir
        self._requested_device = device
        
        # Modèle et processeur (chargés à la demande)
        self._model = None
        self._processor = None
        self._device = None
        self._torch_dtype = None
    
    @property
    def is_available(self) -> bool:
        """
        Vérifie si Florence-2 peut être utilisé.
        
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
        Charge le modèle Florence-2 depuis HuggingFace.
        
        Le modèle est téléchargé automatiquement au premier chargement.
        
        Note:
            Florence-2 nécessite `trust_remote_code=True` car il utilise
            du code personnalisé non inclus dans transformers.
        
        Raises:
            RuntimeError: Si le chargement échoue.
        """
        if self._is_loaded:
            return
        
        import warnings
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        # Ignorer les warnings non critiques
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        # Détecter le device et le dtype
        if self._requested_device:
            self._device = torch.device(self._requested_device)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Utiliser float16 sur GPU, float32 sur CPU
        if self._device.type == "cuda":
            self._torch_dtype = torch.float16
        else:
            self._torch_dtype = torch.float32
        
        # Charger le processeur
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
        
        # Charger le modèle avec attn_implementation="eager" pour éviter les problèmes SDPA
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=self._torch_dtype,
                trust_remote_code=True,
                attn_implementation="eager",  # Éviter les problèmes SDPA
            )
        except TypeError:
            # Fallback si attn_implementation n'est pas supporté (anciennes versions)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=self._torch_dtype,
                trust_remote_code=True,
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
        Résout un CAPTCHA avec Florence-2 (zero-shot OCR).
        
        Args:
            image_bytes: Image au format PNG/JPG en bytes.
            
        Returns:
            SolverResult avec le texte prédit.
        """
        start_time = time.time()
        
        # Charger le modèle si nécessaire
        self.ensure_loaded()
        
        import torch
        
        # Charger l'image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Prétraitement
        inputs = self._processor(
            text=self.OCR_PROMPT,
            images=image,
            return_tensors="pt",
        )
        
        # Déplacer vers le device avec le bon dtype
        inputs = {
            k: v.to(self._device, self._torch_dtype) if v.dtype == torch.float32 else v.to(self._device)
            for k, v in inputs.items()
        }
        
        # Génération
        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=64,
                num_beams=3,
                do_sample=False,
            )
        
        # Décoder le résultat
        generated_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]
        
        # Extraire le texte OCR du résultat
        text = self._parse_ocr_result(generated_text)
        
        processing_time = (time.time() - start_time) * 1000
        
        return SolverResult(
            text=text,
            confidence=None,  # Florence-2 ne fournit pas de confiance directe
            model=self.name,
            processing_time_ms=round(processing_time, 2),
            metadata={
                "model_name": self.model_name,
                "device": str(self._device),
                "prompt": self.OCR_PROMPT,
                "raw_output": generated_text[:200],  # Limiter pour le debug
            },
        )
    
    def _parse_ocr_result(self, raw_output: str) -> str:
        """
        Parse le résultat brut de Florence-2 pour extraire le texte OCR.
        
        Florence-2 retourne le résultat dans un format spécifique
        qu'il faut parser pour extraire uniquement le texte.
        
        Args:
            raw_output: Sortie brute du modèle.
            
        Returns:
            Texte OCR nettoyé.
        """
        # Retirer les tokens spéciaux
        text = raw_output
        
        # Patterns à retirer
        patterns_to_remove = [
            r"<s>",
            r"</s>",
            r"<pad>",
            r"<OCR>",
            r"</OCR>",
            r"<s_ocr>",
            r"</s_ocr>",
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        
        # Nettoyer les espaces
        text = text.strip()
        
        # Pour les CAPTCHAs, on garde seulement les caractères alphanumériques
        # car Florence-2 peut parfois ajouter de la ponctuation
        text = re.sub(r"[^a-zA-Z0-9]", "", text)
        
        return text
    
    def solve_with_prompt(
        self,
        image_bytes: bytes,
        custom_prompt: str,
    ) -> SolverResult:
        """
        Résout avec un prompt personnalisé.
        
        Utile pour des cas spécifiques où le prompt OCR standard
        ne fonctionne pas bien.
        
        Args:
            image_bytes: Image en bytes.
            custom_prompt: Prompt personnalisé.
            
        Returns:
            SolverResult avec le texte prédit.
        """
        start_time = time.time()
        
        self.ensure_loaded()
        
        import torch
        
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        inputs = self._processor(
            text=custom_prompt,
            images=image,
            return_tensors="pt",
        )
        
        inputs = {
            k: v.to(self._device, self._torch_dtype) if v.dtype == torch.float32 else v.to(self._device)
            for k, v in inputs.items()
        }
        
        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=64,
                num_beams=3,
            )
        
        generated_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        
        processing_time = (time.time() - start_time) * 1000
        
        return SolverResult(
            text=generated_text.strip(),
            confidence=None,
            model=self.name,
            processing_time_ms=round(processing_time, 2),
            metadata={
                "custom_prompt": custom_prompt,
            },
        )
    
    def unload(self) -> None:
        """Libère la mémoire du modèle."""
        import torch
        
        if self._model is not None:
            del self._model
            self._model = None
        
        if self._processor is not None:
            del self._processor
            self._processor = None
        
        # Libérer la mémoire GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._is_loaded = False


class FlorenceSolverLarge(FlorenceSolver):
    """
    Version avec le modèle Florence-2-large.
    
    Plus précis mais plus lent et gourmand en mémoire.
    Recommandé uniquement si un GPU est disponible.
    """
    
    DEFAULT_MODEL = "microsoft/Florence-2-large"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._name = "florence_large"
        if "model_name" not in kwargs:
            self.model_name = self.DEFAULT_MODEL