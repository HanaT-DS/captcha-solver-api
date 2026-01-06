"""
Services package for CAPTCHA Factory.

Contient tous les services métier:
- CaptchaGenerator : Génération de CAPTCHAs
- SolverService : Résolution multi-modèle
- ScraperService : Webscraping avec bypass CAPTCHA

M2 MoSEF - Université Paris 1 Panthéon-Sorbonne
"""

from app.services.captcha_generator import (
    CaptchaGenerator,
    GeneratorConfig,
    CharsetType,
    CHARSETS,
    DEFAULT_CHARSET,
)

from app.services.solver_service import (
    SolverService,
    EasyOCRSolver,
    SolverResult,
    CompareResult,
    BenchmarkResult,
)

# Import conditionnel du scraper (nécessite playwright)
try:
    from app.services.scraper_service import ScraperService, scrape_sync
    _SCRAPER_AVAILABLE = True
except ImportError:
    ScraperService = None
    scrape_sync = None
    _SCRAPER_AVAILABLE = False

__all__ = [
    # Generator
    "CaptchaGenerator",
    "GeneratorConfig",
    "CharsetType",
    "CHARSETS",
    "DEFAULT_CHARSET",
    # Solver
    "SolverService",
    "EasyOCRSolver",
    "SolverResult",
    "CompareResult",
    "BenchmarkResult",
    # Scraper
    "ScraperService",
    "scrape_sync",
]