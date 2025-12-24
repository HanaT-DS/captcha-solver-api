from pathlib import Path

# Racine du projet (…/project_root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dossier des images CAPTCHA
DATA_DIR = PROJECT_ROOT / "data" / "captcha_images_v2"
BATCH_SIZE = 4         
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 75
NUM_WORKERS = 0        
EPOCHS = 50             
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"         