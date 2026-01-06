"""
CAPTCHA Generator Service
=========================
Genere des CAPTCHAs avec parametres personnalisables.

M2 MoSEF - Universite Paris 1 Pantheon-Sorbonne
"""

import random
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


DEFAULT_CHARSET = "23456789bcdefgmnpwxy"


class CaptchaGenerator:
    """
    Generateur de CAPTCHAs avec parametres personnalisables.
    
    Attributes:
        charset: Caracteres utilises pour generer le texte
    """
    
    def __init__(self, charset: Optional[str] = None):
        """
        Initialise le generateur.
        
        Args:
            charset: Caracteres a utiliser. Par defaut, charset du dataset original.
        """
        self.charset = charset or DEFAULT_CHARSET
    
    def generate(
        self,
        length: int = 5,
        width: int = 200,
        height: int = 60,
        noise_level: float = 0.3,
    ) -> Tuple[Image.Image, str]:
        """
        Genere une image CAPTCHA.
        
        Args:
            length: Nombre de caracteres
            width: Largeur de l'image en pixels
            height: Hauteur de l'image en pixels
            noise_level: Intensite du bruit (0 a 1)
        
        Returns:
            Tuple (image PIL, texte du CAPTCHA)
        """
        text = self._generate_text(length)
        image = self._create_image(text, width, height, noise_level)
        return image, text
    
    def generate_batch(
        self,
        n: int,
        length: int = 5,
        width: int = 200,
        height: int = 60,
        noise_level: float = 0.3,
    ) -> List[Tuple[Image.Image, str]]:
        """
        Genere plusieurs CAPTCHAs.
        
        Args:
            n: Nombre de CAPTCHAs a generer
        
        Returns:
            Liste de tuples (image, texte)
        """
        return [self.generate(length, width, height, noise_level) for _ in range(n)]
    
    def _generate_text(self, length: int) -> str:
        """Genere un texte aleatoire."""
        return "".join(random.choices(self.charset, k=length))
    
    def _create_image(
        self, 
        text: str, 
        width: int, 
        height: int, 
        noise_level: float
    ) -> Image.Image:
        """Cree l'image CAPTCHA avec le texte et les distorsions."""
        # Couleur de fond aleatoire (claire)
        bg_color = self._random_light_color()
        image = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(image)
        
        # Charger la police
        font = self._load_font(int(height * 0.7))
        
        # Dessiner chaque caractere
        char_width = width // (len(text) + 1)
        
        for i, char in enumerate(text):
            char_color = self._random_dark_color()
            x = char_width * (i + 0.5) + random.randint(-5, 5)
            y = (height - int(height * 0.7)) // 2 + random.randint(-5, 5)
            draw.text((x, y), char, font=font, fill=char_color)
        
        # Ajouter le bruit
        if noise_level > 0:
            image = self._add_noise_lines(image, noise_level)
        
        if noise_level > 0.2:
            image = self._add_noise_dots(image, noise_level)
        
        if noise_level > 0.3:
            image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return image
    
    def _load_font(self, size: int):
        """Charge une police TrueType ou retourne la police par defaut."""
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/verdana.ttf",
            "C:/Windows/Fonts/consola.ttf",
        ]
        
        for path in font_paths:
            try:
                return ImageFont.truetype(path, size)
            except (OSError, IOError):
                continue
        
        return ImageFont.load_default()
    
    def _random_light_color(self) -> Tuple[int, int, int]:
        """Genere une couleur claire pour le fond."""
        return (
            random.randint(200, 255),
            random.randint(200, 255),
            random.randint(200, 255),
        )
    
    def _random_dark_color(self) -> Tuple[int, int, int]:
        """Genere une couleur sombre pour le texte."""
        return (
            random.randint(0, 100),
            random.randint(0, 100),
            random.randint(0, 100),
        )
    
    def _add_noise_lines(self, image: Image.Image, level: float) -> Image.Image:
        """Ajoute des lignes aleatoires a l'image."""
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        n_lines = int(level * 8)
        for _ in range(n_lines):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = random.randint(0, width)
            y2 = random.randint(0, height)
            color = self._random_dark_color()
            draw.line([(x1, y1), (x2, y2)], fill=color, width=1)
        
        return image
    
    def _add_noise_dots(self, image: Image.Image, level: float) -> Image.Image:
        """Ajoute des points de bruit a l'image."""
        img_array = np.array(image)
        
        n_pixels = int(level * 500)
        for _ in range(n_pixels):
            x = random.randint(0, img_array.shape[1] - 1)
            y = random.randint(0, img_array.shape[0] - 1)
            
            if random.random() > 0.5:
                img_array[y, x] = [0, 0, 0]
            else:
                img_array[y, x] = [255, 255, 255]
        
        return Image.fromarray(img_array)
