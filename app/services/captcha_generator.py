"""
CAPTCHA Generator Service - Générateur de CAPTCHAs amélioré.

Génère des images CAPTCHA avec paramètres personnalisables:
- Longueur variable (1-10 caractères)
- Dimensions personnalisables
- Plusieurs niveaux de bruit
- Support de plusieurs charsets

M2 MoSEF - Université Paris 1 Panthéon-Sorbonne
"""

import random
from typing import Tuple, List, Optional, Literal
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


# =============================================================================
# CHARSETS PRÉDÉFINIS
# =============================================================================

class CharsetType(Enum):
    """Types de charsets disponibles."""
    CRNN = "crnn"           # 19 caractères (dataset original)
    DIGITS = "digits"       # 0-9
    LOWERCASE = "lowercase" # a-z
    UPPERCASE = "uppercase" # A-Z
    ALPHANUMERIC = "alphanumeric"  # a-z + A-Z + 0-9
    FULL = "full"           # Tout inclus


# Définitions des charsets
CHARSETS = {
    CharsetType.CRNN: "23456789bcdefgmnpwxy",
    CharsetType.DIGITS: "0123456789",
    CharsetType.LOWERCASE: "abcdefghijklmnopqrstuvwxyz",
    CharsetType.UPPERCASE: "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    CharsetType.ALPHANUMERIC: "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    CharsetType.FULL: "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
}

# Charset par défaut (compatible CRNN de Hana)
DEFAULT_CHARSET = CHARSETS[CharsetType.CRNN]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class GeneratorConfig:
    """Configuration du générateur."""
    
    # Dimensions par défaut
    default_width: int = 200
    default_height: int = 60
    default_length: int = 5
    default_noise_level: float = 0.3
    
    # Limites
    min_length: int = 1
    max_length: int = 10
    min_width: int = 100
    max_width: int = 400
    min_height: int = 40
    max_height: int = 150
    
    # Couleurs
    background_min: int = 200
    background_max: int = 255
    text_min: int = 0
    text_max: int = 100


# =============================================================================
# GÉNÉRATEUR
# =============================================================================

class CaptchaGenerator:
    """
    Générateur de CAPTCHAs avec paramètres personnalisables.
    
    Attributes:
        charset: Caractères utilisés pour générer le texte.
        config: Configuration du générateur.
    """
    
    def __init__(
        self,
        charset: Optional[str] = None,
        charset_type: Optional[CharsetType] = None,
        config: Optional[GeneratorConfig] = None,
    ) -> None:
        """
        Initialise le générateur.
        
        Args:
            charset: Caractères personnalisés (prioritaire).
            charset_type: Type de charset prédéfini.
            config: Configuration personnalisée.
        """
        # Déterminer le charset
        if charset:
            self.charset = charset
        elif charset_type:
            self.charset = CHARSETS.get(charset_type, DEFAULT_CHARSET)
        else:
            self.charset = DEFAULT_CHARSET
        
        self.config = config or GeneratorConfig()
        
        # Cache des polices
        self._font_cache = {}
    
    def generate(
        self,
        length: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        noise_level: Optional[float] = None,
        charset: Optional[str] = None,
    ) -> Tuple[Image.Image, str]:
        """
        Génère une image CAPTCHA.
        
        Args:
            length: Nombre de caractères (défaut: 5).
            width: Largeur en pixels (défaut: 200).
            height: Hauteur en pixels (défaut: 60).
            noise_level: Intensité du bruit 0-1 (défaut: 0.3).
            charset: Charset personnalisé pour cette génération.
            
        Returns:
            Tuple (image PIL, texte du CAPTCHA).
        """
        # Paramètres avec valeurs par défaut
        length = length or self.config.default_length
        width = width or self.config.default_width
        height = height or self.config.default_height
        noise_level = noise_level if noise_level is not None else self.config.default_noise_level
        chars = charset or self.charset
        
        # Valider les paramètres
        length = max(self.config.min_length, min(length, self.config.max_length))
        width = max(self.config.min_width, min(width, self.config.max_width))
        height = max(self.config.min_height, min(height, self.config.max_height))
        noise_level = max(0.0, min(1.0, noise_level))
        
        # Générer le texte et l'image
        text = self._generate_text(length, chars)
        image = self._create_image(text, width, height, noise_level)
        
        return image, text
    
    def generate_batch(
        self,
        n: int,
        length: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        noise_level: Optional[float] = None,
        variable_length: bool = False,
        variable_noise: bool = False,
    ) -> List[Tuple[Image.Image, str]]:
        """
        Génère plusieurs CAPTCHAs.
        
        Args:
            n: Nombre de CAPTCHAs à générer.
            length: Longueur fixe (ou longueur max si variable_length=True).
            width: Largeur en pixels.
            height: Hauteur en pixels.
            noise_level: Niveau de bruit (ou max si variable_noise=True).
            variable_length: Si True, longueur aléatoire entre 3 et length.
            variable_noise: Si True, bruit aléatoire entre 0 et noise_level.
            
        Returns:
            Liste de tuples (image, texte).
        """
        results = []
        
        for _ in range(n):
            # Variation de la longueur
            current_length = length
            if variable_length and length:
                current_length = random.randint(3, length)
            
            # Variation du bruit
            current_noise = noise_level
            if variable_noise and noise_level:
                current_noise = random.uniform(0, noise_level)
            
            image, text = self.generate(
                length=current_length,
                width=width,
                height=height,
                noise_level=current_noise,
            )
            results.append((image, text))
        
        return results
    
    def generate_for_model(
        self,
        model: Literal["crnn", "trocr", "florence"] = "crnn",
        n: int = 1,
    ) -> List[Tuple[Image.Image, str]]:
        """
        Génère des CAPTCHAs optimisés pour un modèle spécifique.
        
        Args:
            model: Nom du modèle cible.
            n: Nombre de CAPTCHAs.
            
        Returns:
            Liste de tuples (image, texte).
        """
        if model == "crnn":
            # Paramètres du dataset d'entraînement de Hana
            return self.generate_batch(
                n=n,
                length=5,
                width=300,
                height=75,
                noise_level=0.3,
            )
        elif model == "trocr":
            # TrOCR peut gérer des images de tailles variables
            return self.generate_batch(
                n=n,
                length=5,
                width=200,
                height=60,
                noise_level=0.3,
                variable_length=True,
            )
        else:
            # Florence-2 peut tout gérer
            return self.generate_batch(
                n=n,
                variable_length=True,
                variable_noise=True,
            )
    
    def _generate_text(self, length: int, chars: str) -> str:
        """Génère un texte aléatoire."""
        return "".join(random.choices(chars, k=length))
    
    def _create_image(
        self,
        text: str,
        width: int,
        height: int,
        noise_level: float,
    ) -> Image.Image:
        """Crée l'image CAPTCHA avec le texte et les distorsions."""
        
        # Couleur de fond aléatoire (claire)
        bg_color = self._random_light_color()
        image = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(image)
        
        # Charger la police
        font_size = int(height * 0.7)
        font = self._load_font(font_size)
        
        # Calculer l'espacement des caractères
        char_width = width // (len(text) + 1)
        
        # Dessiner chaque caractère
        for i, char in enumerate(text):
            char_color = self._random_dark_color()
            
            # Position avec variation aléatoire
            x = char_width * (i + 0.5) + random.randint(-5, 5)
            y = (height - font_size) // 2 + random.randint(-5, 5)
            
            draw.text((x, y), char, font=font, fill=char_color)
        
        # Ajouter le bruit
        if noise_level > 0:
            image = self._add_noise_lines(image, noise_level)
        
        if noise_level > 0.2:
            image = self._add_noise_dots(image, noise_level)
        
        if noise_level > 0.3:
            image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        if noise_level > 0.5:
            image = self._add_distortion(image, noise_level)
        
        return image
    
    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Charge une police TrueType avec cache."""
        
        # Vérifier le cache
        if size in self._font_cache:
            return self._font_cache[size]
        
        # Chemins de polices à essayer
        font_paths = [
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
            # Windows
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/verdana.ttf",
            "C:/Windows/Fonts/consola.ttf",
            # macOS
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Arial.ttf",
        ]
        
        for path in font_paths:
            try:
                font = ImageFont.truetype(path, size)
                self._font_cache[size] = font
                return font
            except (OSError, IOError):
                continue
        
        # Fallback sur la police par défaut
        return ImageFont.load_default()
    
    def _random_light_color(self) -> Tuple[int, int, int]:
        """Génère une couleur claire pour le fond."""
        return (
            random.randint(self.config.background_min, self.config.background_max),
            random.randint(self.config.background_min, self.config.background_max),
            random.randint(self.config.background_min, self.config.background_max),
        )
    
    def _random_dark_color(self) -> Tuple[int, int, int]:
        """Génère une couleur sombre pour le texte."""
        return (
            random.randint(self.config.text_min, self.config.text_max),
            random.randint(self.config.text_min, self.config.text_max),
            random.randint(self.config.text_min, self.config.text_max),
        )
    
    def _add_noise_lines(self, image: Image.Image, level: float) -> Image.Image:
        """Ajoute des lignes de bruit."""
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        n_lines = int(level * 8)
        for _ in range(n_lines):
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            color = self._random_dark_color()
            draw.line([(x1, y1), (x2, y2)], fill=color, width=1)
        
        return image
    
    def _add_noise_dots(self, image: Image.Image, level: float) -> Image.Image:
        """Ajoute des points de bruit."""
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
    
    def _add_distortion(self, image: Image.Image, level: float) -> Image.Image:
        """Ajoute une distorsion légère."""
        # Simple distorsion par étirement
        width, height = image.size
        
        # Variation aléatoire de la taille
        new_width = width + random.randint(-int(level * 10), int(level * 10))
        new_height = height + random.randint(-int(level * 5), int(level * 5))
        
        # Redimensionner puis revenir à la taille originale
        image = image.resize((new_width, new_height), Image.BILINEAR)
        image = image.resize((width, height), Image.BILINEAR)
        
        return image
    
    @property
    def charset_info(self) -> dict:
        """Informations sur le charset actuel."""
        return {
            "charset": self.charset,
            "size": len(self.charset),
            "has_digits": any(c.isdigit() for c in self.charset),
            "has_lowercase": any(c.islower() for c in self.charset),
            "has_uppercase": any(c.isupper() for c in self.charset),
        }