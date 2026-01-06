"""
Logging Configuration - Configuration centralisée des logs.

M2 MoSEF - Université Paris 1 Panthéon-Sorbonne
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


# =============================================================================
# CONSTANTES
# =============================================================================

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE = "captcha_factory.log"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5


# =============================================================================
# CONFIGURATION
# =============================================================================

def setup_logging(
    level: str = None,
    log_file: str = None,
    log_to_console: bool = True,
    log_to_file: bool = False,
) -> logging.Logger:
    """
    Configure le logging pour l'application.
    
    Args:
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Chemin du fichier de log (optionnel).
        log_to_console: Activer les logs console.
        log_to_file: Activer les logs fichier.
        
    Returns:
        Logger racine configuré.
    """
    # Niveau depuis l'environnement ou paramètre
    level = level or os.getenv("LOG_LEVEL", "INFO")
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Logger racine
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Supprimer les handlers existants
    root_logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    
    # Handler console
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Handler fichier
    if log_to_file:
        log_path = log_file or LOG_FILE
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT,
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Réduire le niveau de certaines bibliothèques
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Récupère un logger avec le nom spécifié.
    
    Args:
        name: Nom du logger (généralement __name__).
        
    Returns:
        Logger configuré.
    """
    return logging.getLogger(name)


# =============================================================================
# FILTRES PERSONNALISÉS
# =============================================================================

class ModelLoadingFilter(logging.Filter):
    """Filtre les messages de chargement de modèles répétitifs."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Filtrer les messages de téléchargement HuggingFace
        if "Downloading" in record.getMessage():
            return False
        if "Loading" in record.getMessage() and "checkpoint" in record.getMessage():
            return False
        return True


# =============================================================================
# DÉCORATEURS UTILES
# =============================================================================

def log_execution_time(func):
    """Décorateur pour logger le temps d'exécution."""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} exécuté en {elapsed:.3f}s")
        return result
    
    return wrapper


def log_exceptions(func):
    """Décorateur pour logger les exceptions."""
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Exception dans {func.__name__}: {e}")
            raise
    
    return wrapper