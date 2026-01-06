"""
Image Processing Utilities - Prétraitement d'images pour les modèles.

Fonctions pour:
- Chargement d'images depuis bytes
- Prétraitement spécifique par modèle
- Conversion base64
- Normalisation

M2 MoSEF - Université Paris 1 Panthéon-Sorbonne
"""

import io
import base64
from typing import Tuple, Optional, Union

import numpy as np
from PIL import Image


# =============================================================================
# CONSTANTES
# =============================================================================

# Normalisation ImageNet (utilisée par tous les modèles)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Dimensions par défaut
CRNN_INPUT_SIZE = (300, 75)  # width x height
TROCR_INPUT_SIZE = (384, 384)  # Carré pour ViT


# =============================================================================
# CHARGEMENT
# =============================================================================

def load_image_from_bytes(
    image_bytes: bytes,
    mode: str = "RGB",
) -> Image.Image:
    """
    Charge une image depuis des bytes.
    
    Args:
        image_bytes: Données de l'image (PNG, JPG, etc.).
        mode: Mode de conversion ('RGB', 'L', 'RGBA').
        
    Returns:
        Image PIL.
        
    Raises:
        ValueError: Si les bytes ne sont pas une image valide.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert(mode)
    except Exception as e:
        raise ValueError(f"Impossible de charger l'image: {e}")


def load_image_from_path(
    path: str,
    mode: str = "RGB",
) -> Image.Image:
    """
    Charge une image depuis un chemin fichier.
    
    Args:
        path: Chemin vers le fichier image.
        mode: Mode de conversion.
        
    Returns:
        Image PIL.
    """
    image = Image.open(path)
    return image.convert(mode)


# =============================================================================
# CONVERSION BASE64
# =============================================================================

def image_to_base64(
    image: Image.Image,
    format: str = "PNG",
) -> str:
    """
    Convertit une image PIL en string base64.
    
    Args:
        image: Image PIL.
        format: Format de sortie ('PNG', 'JPEG').
        
    Returns:
        String base64.
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(
    base64_str: str,
    mode: str = "RGB",
) -> Image.Image:
    """
    Convertit une string base64 en image PIL.
    
    Args:
        base64_str: String base64 (avec ou sans préfixe data:).
        mode: Mode de conversion.
        
    Returns:
        Image PIL.
    """
    # Retirer le préfixe data:image si présent
    if base64_str.startswith("data:"):
        base64_str = base64_str.split(",")[1]
    
    image_bytes = base64.b64decode(base64_str)
    return load_image_from_bytes(image_bytes, mode)


def image_to_bytes(
    image: Image.Image,
    format: str = "PNG",
) -> bytes:
    """
    Convertit une image PIL en bytes.
    
    Args:
        image: Image PIL.
        format: Format de sortie.
        
    Returns:
        Bytes de l'image.
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


# =============================================================================
# REDIMENSIONNEMENT
# =============================================================================

def resize_image(
    image: Image.Image,
    size: Tuple[int, int],
    resample: int = Image.BILINEAR,
    maintain_aspect: bool = False,
) -> Image.Image:
    """
    Redimensionne une image.
    
    Args:
        image: Image PIL.
        size: Tuple (width, height).
        resample: Méthode de rééchantillonnage.
        maintain_aspect: Si True, maintient le ratio avec padding.
        
    Returns:
        Image redimensionnée.
    """
    if not maintain_aspect:
        return image.resize(size, resample)
    
    # Calculer le ratio
    target_w, target_h = size
    orig_w, orig_h = image.size
    
    ratio = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)
    
    # Redimensionner
    image = image.resize((new_w, new_h), resample)
    
    # Ajouter padding
    result = Image.new(image.mode, size, (255, 255, 255))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    result.paste(image, (paste_x, paste_y))
    
    return result


# =============================================================================
# NORMALISATION
# =============================================================================

def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    """
    Normalise une image avec mean/std.
    
    Args:
        image: Array numpy (H, W, C) avec valeurs [0, 255].
        mean: Moyenne par canal.
        std: Écart-type par canal.
        
    Returns:
        Array normalisé.
    """
    image = image.astype(np.float32) / 255.0
    
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    image = (image - mean) / std
    
    return image


def denormalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    """
    Dénormalise une image.
    
    Args:
        image: Array normalisé.
        mean: Moyenne utilisée.
        std: Écart-type utilisé.
        
    Returns:
        Array avec valeurs [0, 255].
    """
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    image = image * std + mean
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    return image


# =============================================================================
# PRÉTRAITEMENT SPÉCIFIQUE PAR MODÈLE
# =============================================================================

def preprocess_for_crnn(
    image: Union[Image.Image, bytes],
    target_size: Tuple[int, int] = CRNN_INPUT_SIZE,
) -> np.ndarray:
    """
    Prétraite une image pour le modèle CRNN.
    
    Args:
        image: Image PIL ou bytes.
        target_size: Taille cible (width, height).
        
    Returns:
        Array numpy (C, H, W) normalisé, prêt pour PyTorch.
    """
    # Charger si bytes
    if isinstance(image, bytes):
        image = load_image_from_bytes(image)
    
    # Redimensionner
    image = resize_image(image, target_size)
    
    # Convertir en array
    image_np = np.array(image)
    
    # Normaliser
    image_np = normalize_image(image_np)
    
    # Réorganiser: (H, W, C) -> (C, H, W)
    image_np = image_np.transpose(2, 0, 1)
    
    return image_np


def preprocess_for_trocr(
    image: Union[Image.Image, bytes],
) -> Image.Image:
    """
    Prétraite une image pour TrOCR.
    
    Note: TrOCR utilise son propre processor, cette fonction
    ne fait qu'un prétraitement minimal.
    
    Args:
        image: Image PIL ou bytes.
        
    Returns:
        Image PIL en RGB.
    """
    if isinstance(image, bytes):
        image = load_image_from_bytes(image)
    
    return image.convert("RGB")


def preprocess_for_florence(
    image: Union[Image.Image, bytes],
) -> Image.Image:
    """
    Prétraite une image pour Florence-2.
    
    Args:
        image: Image PIL ou bytes.
        
    Returns:
        Image PIL en RGB.
    """
    if isinstance(image, bytes):
        image = load_image_from_bytes(image)
    
    return image.convert("RGB")


# =============================================================================
# AUGMENTATION (pour l'entraînement)
# =============================================================================

def get_training_transforms():
    """
    Retourne les transformations pour l'entraînement.
    
    Utilise Albumentations pour des augmentations robustes.
    
    Returns:
        Pipeline d'augmentation Albumentations.
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        return A.Compose([
            # Géométrique
            A.Rotate(limit=5, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=5,
                p=0.5,
            ),
            A.Perspective(scale=(0.02, 0.05), p=0.3),
            
            # Bruit
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.ISONoise(p=0.2),
            
            # Flou
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5)),
                A.MotionBlur(blur_limit=3),
            ], p=0.3),
            
            # Couleur
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            
            # Normalisation
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    
    except ImportError:
        return None


def get_inference_transforms():
    """
    Retourne les transformations pour l'inférence.
    
    Returns:
        Pipeline de transformation Albumentations.
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        return A.Compose([
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    
    except ImportError:
        return None