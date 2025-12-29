"""
Module d'inférence pour la résolution de CAPTCHAs.
"""
import io
import torch
import numpy as np
import albumentations
from PIL import Image
from typing import Dict, List

# Import de la configuration
from app.config import (
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    MEAN,
    STD
)

# Import de l'architecture depuis le fichier model.py
from app.model import CaptchaModel 


def load_trained_model(model_path: str, num_chars: int = 19, device: str = "cpu") -> CaptchaModel:
    """
    Charge l'architecture et les poids du modèle.
    
    Args:
        model_path: Chemin vers le fichier .pth contenant les poids.
        num_chars: Nombre de caractères uniques dans le vocabulaire.
        device: Device cible ('cpu' ou 'cuda').
    
    Returns:
        Le modèle chargé et prêt pour l'inférence.
    """
    model = CaptchaModel(num_chars=num_chars)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device), weights_only=True))
    model.to(device)
    model.eval()
    return model


def decode_predictions(preds: torch.Tensor, mapping: Dict[int, str]) -> List[str]:
    """
    Transforme les sorties numériques du modèle en texte (logique CTC).
    
    Algorithme CTC Greedy correct :
    1. Collapse les répétitions consécutives (sur séquence brute avec blanks)
    2. Supprime les blanks
    
    Args:
        preds: Tensor de prédictions brutes du modèle.
        mapping: Dictionnaire {index: caractère}.
    
    Returns:
        Liste des textes décodés.
    """
    # (T, B, C) → (B, T, C)
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, dim=2)
    preds = torch.argmax(preds, dim=2)
    preds = preds.detach().cpu().numpy()
    
    results: List[str] = []
    blank_idx = 0  # Index du blank dans CTC
    
    for batch_idx in range(preds.shape[0]):
        sequence = preds[batch_idx]
        
        # Décodage CTC greedy correct
        decoded_chars: List[str] = []
        prev_idx = blank_idx  # Commence avec blank
        
        for idx in sequence:
            # Ajouter le caractère si :
            # 1. Ce n'est pas un blank ET
            # 2. C'est différent du précédent (collapse répétitions)
            if idx != blank_idx and idx != prev_idx:
                decoded_chars.append(mapping[idx])
            
            prev_idx = idx
        
        results.append("".join(decoded_chars))
    
    return results


def solve_captcha(image_bytes: bytes, model: torch.nn.Module, mapping: Dict[int, str]) -> str:
    """
    Prépare l'image et lance la prédiction.
    
    Args:
        image_bytes: Contenu binaire de l'image.
        model: Modèle PyTorch chargé.
        mapping: Dictionnaire {index: caractère}.
    
    Returns:
        Le texte prédit du CAPTCHA.
    """
    # 1. Conversion des octets en image PIL
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 2. Prétraitement (Redimensionnement & Normalisation)
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), resample=Image.BILINEAR)
    image_array = np.array(image)
    
    aug = albumentations.Compose([
        albumentations.Normalize(
            mean=MEAN, 
            std=STD
        )
    ])
    
    image_norm = aug(image=image_array)["image"]
    image_tensor = np.transpose(image_norm, (2, 0, 1)).astype(np.float32)
    image_tensor = torch.tensor(image_tensor, dtype=torch.float).unsqueeze(0)
    
    # 3. Inférence
    with torch.no_grad():
        preds, _ = model(image_tensor)
        
    # 4. Décodage
    decoded = decode_predictions(preds, mapping)
    return decoded[0]