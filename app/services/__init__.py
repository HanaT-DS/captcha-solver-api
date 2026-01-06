"""
Services package for CAPTCHA Factory.
"""

from app.services.captcha_generator import CaptchaGenerator
from app.services.solver_service import SolverService, EasyOCRSolver, CRNNSolver

__all__ = [
    "CaptchaGenerator",
    "SolverService",
    "EasyOCRSolver",
    "CRNNSolver",
]

# ScraperService est importe separement car il necessite Playwright
# from app.services.scraper_service import ScraperService
