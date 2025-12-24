import albumentations              # Librairie de data augmentation / normalisation
import torch                       # PyTorch
import numpy as np                 

from PIL import Image              # Lecture des images
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Évite les erreurs sur images corrompues


class ClassificationDataset:
    def __init__(self, image_paths, targets, resize=None):
        """
        image_paths : liste des chemins des images
        targets     : labels associés aux images
        resize      : tuple (height, width)
        """
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize

        # Moyenne et écart-type standards (ImageNet)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        # Pipeline de normalisation
        self.aug = albumentations.Compose(
            [
                albumentations.Normalize(
                    mean=mean,
                    std=std,
                    max_pixel_value=255.0,
                    always_apply=True
                )
            ]
        )

    def __len__(self):
        # Nombre total d'images
        return len(self.image_paths)

    def __getitem__(self, item):
        # Ouvre l'image et force le format RGB
        image = Image.open(self.image_paths[item]).convert("RGB")
        targets = self.targets[item]

        # Resize si demandé
        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]),  # (width, height)
                resample=Image.BILINEAR
            )

        # Conversion PIL -> NumPy
        image = np.array(image)

        # Normalisation avec albumentations
        augmented = self.aug(image=image)
        image = augmented["image"]

        # (H, W, C) -> (C, H, W) pour PyTorch
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        # Retour sous forme de tensors PyTorch
        return {
            "images": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long),
        }
