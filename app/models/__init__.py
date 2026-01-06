"""
Models package for CAPTCHA Factory.

Contient les définitions de tous les modèles de résolution:
- CRNN : Modèle entraîné par Hana (98% accuracy, charset limité)
- TrOCR : Modèle pré-entraîné HuggingFace (99% accuracy, charset complet)
- Florence-2 : VLM zero-shot (universel)

M2 MoSEF - Université Paris 1 Panthéon-Sorbonne
"""

from app.models.base_solver import (
    BaseSolver,
    SolverResult,
    ModelType,
)

from app.models.crnn_model import CRNNSolver
from app.models.trocr_solver import TrOCRSolver, TrOCRSolverAlternative
from app.models.florence_solver import FlorenceSolver, FlorenceSolverLarge

__all__ = [
    # Base
    "BaseSolver",
    "SolverResult",
    "ModelType",
    # Solvers
    "CRNNSolver",
    "TrOCRSolver",
    "TrOCRSolverAlternative",
    "FlorenceSolver",
    "FlorenceSolverLarge",
]