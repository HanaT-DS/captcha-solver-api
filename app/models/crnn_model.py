"""
CRNN Solver - Modèle CRNN entraîné par Hana.

Architecture:
- CNN : 2 couches convolutionnelles (3→128→64)
- GRU Bidirectionnel : 2 couches, 32 unités cachées
- CTC Loss : Décodage avec greedy search

Performance:
- Accuracy : 98.08% sur le dataset captcha_images_v2
- CER : 0.38%
- Charset : 19 caractères (23456789bcdefgmnpwxy)

M2 MoSEF - Université Paris 1 Panthéon-Sorbonne
"""

import io
import time
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from PIL import Image

from app.models.base_solver import BaseSolver, SolverResult, ModelType


class CRNNSolver(BaseSolver):
    """
    Solver utilisant le modèle CRNN entraîné sur captcha_images_v2.
    
    Ce modèle est optimisé pour les CAPTCHAs avec:
    - 5 caractères
    - Charset limité (19 caractères)
    - Dimensions 300x75 pixels
    
    Attributes:
        model_path: Chemin vers les poids du modèle (.pth).
        mapping_path: Chemin vers le mapping des caractères (.joblib).
    """
    
    # Charset supporté (dataset original)
    CHARSET = "23456789bcdefgmnpwxy"
    
    # Dimensions d'entrée
    IMAGE_WIDTH = 300
    IMAGE_HEIGHT = 75
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        mapping_path: Optional[str] = None,
    ) -> None:
        """
        Initialise le solver CRNN.
        
        Args:
            model_path: Chemin vers le fichier .pth (optionnel).
            mapping_path: Chemin vers le fichier .joblib (optionnel).
        """
        super().__init__(
            name="crnn",
            model_type=ModelType.CRNN,
            charset=self.CHARSET,
        )
        
        # Chemins par défaut
        base_path = Path(__file__).parent.parent.parent
        self.model_path = Path(model_path) if model_path else base_path / "models" / "captcha_model.pth"
        self.mapping_path = Path(mapping_path) if mapping_path else base_path / "models" / "captcha_model_mapping.joblib"
        
        # Modèle et métadonnées (chargés à la demande)
        self._model = None
        self._metadata = None
        self._device = None
    
    @property
    def is_available(self) -> bool:
        """Vérifie si les fichiers du modèle existent."""
        return self.model_path.exists() and self.mapping_path.exists()
    
    def load(self) -> None:
        """
        Charge le modèle CRNN et les métadonnées.
        
        Raises:
            FileNotFoundError: Si les fichiers n'existent pas.
            RuntimeError: Si le chargement échoue.
        """
        if self._is_loaded:
            return
        
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import joblib
        
        # Vérifier les fichiers
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modèle CRNN non trouvé : {self.model_path}")
        
        if not self.mapping_path.exists():
            raise FileNotFoundError(f"Mapping non trouvé : {self.mapping_path}")
        
        # Charger les métadonnées
        self._metadata = joblib.load(self.mapping_path)
        
        # Définir l'architecture (identique à l'entraînement)
        class CaptchaModel(nn.Module):
            """Architecture CRNN pour la reconnaissance de CAPTCHAs."""
            
            def __init__(self, num_chars: int) -> None:
                super().__init__()
                
                # CNN
                self.conv_1 = nn.Conv2d(3, 128, kernel_size=(3, 6), padding=(1, 1))
                self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
                self.conv_2 = nn.Conv2d(128, 64, kernel_size=(3, 6), padding=(1, 1))
                self.pool_2 = nn.MaxPool2d(kernel_size=(2, 2))
                
                # Transition
                self.linear_1 = nn.Linear(1152, 64)
                self.drop_1 = nn.Dropout(0.2)
                
                # RNN (GRU bidirectionnel)
                self.gru = nn.GRU(
                    input_size=64,
                    hidden_size=32,
                    num_layers=2,
                    bidirectional=True,
                    dropout=0.25,
                    batch_first=True,
                )
                
                # Sortie
                self.output = nn.Linear(64, num_chars + 1)  # +1 pour le blank CTC
            
            def forward(self, images: torch.Tensor, targets: Optional[torch.Tensor] = None):
                bs = images.size(0)
                
                # CNN
                x = F.relu(self.conv_1(images))
                x = self.pool_1(x)
                x = F.relu(self.conv_2(x))
                x = self.pool_2(x)
                
                # Reshape pour RNN : (B, C, H, W) -> (B, W, H*C)
                x = x.permute(0, 3, 1, 2)
                x = x.view(bs, x.size(1), -1)
                
                # Linear + Dropout
                x = F.relu(self.linear_1(x))
                x = self.drop_1(x)
                
                # RNN
                x, _ = self.gru(x)
                
                # Sortie : (B, T, num_classes)
                x = self.output(x)
                
                # Format CTC : (T, B, C)
                x = x.permute(1, 0, 2)
                
                return x, None
        
        # Initialiser et charger les poids
        num_chars = len(self._metadata.get("vocab", self.CHARSET))
        self._model = CaptchaModel(num_chars)
        
        # Détecter le device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Charger les poids
        state_dict = torch.load(
            self.model_path,
            map_location=self._device,
            weights_only=True,
        )
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()
        
        self._is_loaded = True
    
    def solve(self, image_bytes: bytes) -> SolverResult:
        """
        Résout un CAPTCHA avec le modèle CRNN.
        
        Args:
            image_bytes: Image au format PNG/JPG en bytes.
            
        Returns:
            SolverResult avec le texte prédit.
        """
        start_time = time.time()
        
        # Charger le modèle si nécessaire
        self.ensure_loaded()
        
        import torch
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        # Charger et prétraiter l'image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((self.IMAGE_WIDTH, self.IMAGE_HEIGHT), Image.BILINEAR)
        image_np = np.array(image)
        
        # Appliquer les transformations (normalisation ImageNet)
        transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
        transformed = transform(image=image_np)
        image_tensor = transformed["image"].unsqueeze(0).to(self._device)
        
        # Inférence
        with torch.no_grad():
            preds, _ = self._model(image_tensor)
        
        # Décoder les prédictions
        text, confidence = self._decode_predictions(preds)
        
        processing_time = (time.time() - start_time) * 1000
        
        return SolverResult(
            text=text,
            confidence=confidence,
            model=self.name,
            processing_time_ms=round(processing_time, 2),
            metadata={
                "input_size": f"{self.IMAGE_WIDTH}x{self.IMAGE_HEIGHT}",
                "charset_size": len(self.CHARSET),
            },
        )
    
    def _decode_predictions(self, preds) -> tuple[str, Optional[float]]:
        """
        Décode les prédictions CTC avec greedy decoding.
        
        Args:
            preds: Tensor de sortie du modèle (T, B, C).
            
        Returns:
            Tuple (texte décodé, confiance moyenne).
        """
        import torch
        
        # Récupérer le mapping
        mapping = self._metadata.get("num_to_char", {})
        
        # (T, B, C) -> (B, T, C)
        preds = preds.permute(1, 0, 2)
        
        # Softmax pour obtenir les probabilités
        probs = torch.softmax(preds, dim=2)
        
        # Max pour obtenir les indices et les scores
        max_probs, indices = torch.max(probs, dim=2)
        
        # Décoder (greedy)
        sequence = indices[0].cpu().numpy()
        prob_sequence = max_probs[0].cpu().numpy()
        
        decoded_chars = []
        char_probs = []
        prev_idx = 0  # Index du blank
        
        for idx, prob in zip(sequence, prob_sequence):
            if idx != 0 and idx != prev_idx:
                if idx in mapping:
                    decoded_chars.append(mapping[idx])
                    char_probs.append(prob)
            prev_idx = idx
        
        text = "".join(decoded_chars)
        
        # Calculer la confiance moyenne
        confidence = float(np.mean(char_probs)) if char_probs else 0.0
        
        return text, round(confidence, 3)
    
    def unload(self) -> None:
        """Libère la mémoire du modèle."""
        import torch
        
        if self._model is not None:
            del self._model
            self._model = None
        
        if self._metadata is not None:
            del self._metadata
            self._metadata = None
        
        # Libérer la mémoire GPU si disponible
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._is_loaded = False