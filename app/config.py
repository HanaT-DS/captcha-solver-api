import torch
from pathlib import Path


# ============================================
# CHEMINS
# ============================================
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "captcha_model.pth"
MAPPING_PATH = MODELS_DIR / "captcha_model_mapping.joblib"


# ============================================
# PARAMÈTRES IMAGE
# ============================================
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 75


# ============================================
# DEVICE
# ============================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================
# NORMALISATION (ImageNet)
# ============================================
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]