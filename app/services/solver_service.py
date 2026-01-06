"""
CAPTCHA Solver Service
======================
Multi-model solver supportant EasyOCR et CRNN.

Modeles disponibles:
- EasyOCR : modele pre-entraine, rapide sur CPU
- CRNN : modele personnalise entraine sur captcha_images_v2

M2 MoSEF - Universite Paris 1 Pantheon-Sorbonne
"""

import io
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from PIL import Image


class EasyOCRSolver:
    """
    Solver utilisant EasyOCR (modele pre-entraine).
    
    EasyOCR est efficace pour la reconnaissance de texte general
    et fonctionne bien sur CPU.
    """
    
    def __init__(self, languages: List[str] = None):
        """
        Initialise le solver EasyOCR.
        
        Args:
            languages: Liste des langues (par defaut ['en'])
        """
        self.languages = languages or ["en"]
        self.reader = None
        self._loaded = False
    
    def load(self) -> None:
        """Charge le modele EasyOCR."""
        if not self._loaded:
            import easyocr
            self.reader = easyocr.Reader(
                self.languages, 
                gpu=False, 
                verbose=False
            )
            self._loaded = True
    
    def solve(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Resout un CAPTCHA a partir des bytes de l'image.
        
        Args:
            image_bytes: Image au format PNG/JPG en bytes
        
        Returns:
            Dictionnaire avec 'text' et 'confidence'
        """
        if not self._loaded:
            self.load()
        
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        
        results = self.reader.readtext(image_np)
        
        if not results:
            return {"text": "", "confidence": 0.0}
        
        text = "".join([r[1] for r in results])
        text = "".join(c for c in text if c.isalnum())
        avg_confidence = sum(r[2] for r in results) / len(results)
        
        return {
            "text": text.lower(),
            "confidence": round(avg_confidence, 3)
        }
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded


class CRNNSolver:
    """
    Solver utilisant le modele CRNN entraine par l'equipe.
    
    Architecture:
    - CNN : extraction de features
    - GRU bidirectionnel : modelisation sequentielle
    - CTC : decodage du texte
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None, 
        mapping_path: Optional[str] = None
    ):
        """
        Initialise le solver CRNN.
        
        Args:
            model_path: Chemin vers le fichier .pth
            mapping_path: Chemin vers le fichier .joblib
        """
        # Chemins par defaut
        base_path = Path(__file__).parent.parent.parent
        self.model_path = model_path or str(base_path / "models" / "captcha_model.pth")
        self.mapping_path = mapping_path or str(base_path / "models" / "captcha_model_mapping.joblib")
        
        self.model = None
        self.metadata = None
        self._loaded = False
    
    def load(self) -> None:
        """Charge le modele CRNN et les metadonnees."""
        if self._loaded:
            return
        
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import joblib
        
        # Verifier que les fichiers existent
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Modele non trouve : {self.model_path}")
        
        if not Path(self.mapping_path).exists():
            raise FileNotFoundError(f"Mapping non trouve : {self.mapping_path}")
        
        # Charger les metadonnees
        self.metadata = joblib.load(self.mapping_path)
        
        # Definir l'architecture du modele (doit correspondre a l'entrainement)
        class CaptchaModel(nn.Module):
            def __init__(self, num_chars: int):
                super().__init__()
                self.conv_1 = nn.Conv2d(3, 128, kernel_size=(3, 6), padding=(1, 1))
                self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
                self.conv_2 = nn.Conv2d(128, 64, kernel_size=(3, 6), padding=(1, 1))
                self.pool_2 = nn.MaxPool2d(kernel_size=(2, 2))
                self.linear_1 = nn.Linear(1152, 64)
                self.drop_1 = nn.Dropout(0.2)
                self.gru = nn.GRU(
                    input_size=64,
                    hidden_size=32,
                    num_layers=2,
                    bidirectional=True,
                    dropout=0.25,
                    batch_first=True
                )
                self.output = nn.Linear(64, num_chars + 1)
            
            def forward(self, images, targets=None):
                bs = images.size(0)
                x = F.relu(self.conv_1(images))
                x = self.pool_1(x)
                x = F.relu(self.conv_2(x))
                x = self.pool_2(x)
                x = x.permute(0, 3, 1, 2)
                x = x.view(bs, x.size(1), -1)
                x = F.relu(self.linear_1(x))
                x = self.drop_1(x)
                x, _ = self.gru(x)
                x = self.output(x)
                x = x.permute(1, 0, 2)
                return x, None
        
        # Initialiser et charger les poids
        num_chars = len(self.metadata["vocab"])
        self.model = CaptchaModel(num_chars)
        self.model.load_state_dict(
            torch.load(self.model_path, map_location="cpu", weights_only=True)
        )
        self.model.eval()
        
        self._loaded = True
    
    def solve(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Resout un CAPTCHA avec le modele CRNN.
        
        Args:
            image_bytes: Image au format PNG/JPG en bytes
        
        Returns:
            Dictionnaire avec 'text' et 'confidence'
        """
        if not self._loaded:
            self.load()
        
        import torch
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        # Charger et preprocesser l'image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Redimensionner a la taille d'entrainement
        image = image.resize((300, 75))
        image_np = np.array(image)
        
        # Appliquer les transformations
        transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
        transformed = transform(image=image_np)
        image_tensor = transformed["image"].unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            preds, _ = self.model(image_tensor)
        
        # Decoder les predictions
        text = self._decode_predictions(preds)
        
        return {
            "text": text,
            "confidence": None  # CTC ne fournit pas de confiance directe
        }
    
    def _decode_predictions(self, preds) -> str:
        """
        Decode les predictions CTC avec greedy decoding.
        
        Args:
            preds: Tensor de sortie du modele (T, B, C)
        
        Returns:
            Texte decode
        """
        import torch
        
        # Recuperer le mapping
        mapping = self.metadata["num_to_char"]
        
        # (T, B, C) -> (B, T, C)
        preds = preds.permute(1, 0, 2)
        preds = torch.softmax(preds, dim=2)
        preds = torch.argmax(preds, dim=2)
        preds = preds.detach().cpu().numpy()
        
        # Decoder le premier (et seul) element du batch
        sequence = preds[0]
        decoded_chars = []
        prev_idx = 0  # Index du blank
        
        for idx in sequence:
            if idx != 0 and idx != prev_idx:
                if idx in mapping:
                    decoded_chars.append(mapping[idx])
            prev_idx = idx
        
        return "".join(decoded_chars)
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded


class SolverService:
    """
    Service de resolution multi-modele.
    
    Gere plusieurs solvers et fournit une interface unifiee.
    Modeles disponibles:
    - easyocr : modele pre-entraine
    - crnn : modele personnalise
    """
    
    def __init__(self):
        """Initialise les solvers disponibles."""
        self.solvers = {
            "easyocr": EasyOCRSolver(),
        }
        
        # Essayer d'ajouter CRNN si les fichiers existent
        try:
            crnn = CRNNSolver()
            # Verifier que les fichiers existent
            if Path(crnn.model_path).exists() and Path(crnn.mapping_path).exists():
                self.solvers["crnn"] = crnn
        except Exception:
            pass
        
        self._default_model = "easyocr"
    
    def solve(
        self,
        image_bytes: bytes,
        model: str = "easyocr"
    ) -> Dict[str, Any]:
        """
        Resout un CAPTCHA avec le modele specifie.
        
        Args:
            image_bytes: Image en bytes
            model: Nom du modele ('easyocr', 'crnn', ou 'all')
        
        Returns:
            Dictionnaire avec 'text', 'model', et optionnellement 'confidence'
        """
        if model == "all":
            return self.solve_all(image_bytes)
        
        if model not in self.solvers:
            model = self._default_model
        
        try:
            result = self.solvers[model].solve(image_bytes)
            result["model"] = model
            return result
        except FileNotFoundError as e:
            # Si le modele n'est pas disponible, utiliser le defaut
            if model != self._default_model:
                result = self.solvers[self._default_model].solve(image_bytes)
                result["model"] = self._default_model
                result["warning"] = f"Modele {model} non disponible, utilisation de {self._default_model}"
                return result
            raise e
    
    def solve_all(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Resout avec tous les modeles disponibles.
        
        Args:
            image_bytes: Image en bytes
        
        Returns:
            Dictionnaire avec les resultats de tous les modeles
        """
        results = {}
        
        for name, solver in self.solvers.items():
            try:
                results[name] = solver.solve(image_bytes)
            except Exception as e:
                results[name] = {"error": str(e)}
        
        # Trouver le meilleur resultat
        best_model = None
        best_result = None
        
        for name, result in results.items():
            if "error" not in result:
                if best_result is None:
                    best_model = name
                    best_result = result
                elif result.get("confidence") and best_result.get("confidence"):
                    if result["confidence"] > best_result["confidence"]:
                        best_model = name
                        best_result = result
        
        return {
            "text": best_result["text"] if best_result else "",
            "model": best_model or "none",
            "all_results": results
        }
    
    def get_available_models(self) -> List[str]:
        """Retourne la liste des modeles disponibles."""
        return list(self.solvers.keys())
    
    def get_loaded_models(self) -> List[str]:
        """Retourne la liste des modeles charges en memoire."""
        return [name for name, solver in self.solvers.items() if solver.is_loaded]
