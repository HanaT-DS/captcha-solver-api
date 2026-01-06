"""
CAPTCHA Solver Service - Service de résolution multi-modèle.

Ce service orchestre plusieurs modèles de résolution et fournit:
- Résolution simple avec un modèle spécifique
- Résolution en cascade (CRNN → TrOCR → Florence-2)
- Comparaison des modèles
- Benchmark automatique

Architecture de cascade:
1. CRNN : Rapide, utilisé si charset compatible (19 chars)
2. TrOCR : Très précis (99%), charset complet
3. Florence-2 : Fallback universel (zero-shot VLM)

M2 MoSEF - Université Paris 1 Panthéon-Sorbonne
"""

import io
import time
import logging
from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from app.models.base_solver import BaseSolver, SolverResult, ModelType
from app.models.crnn_model import CRNNSolver
from app.models.trocr_solver import TrOCRSolver
from app.models.florence_solver import FlorenceSolver

# Logger
logger = logging.getLogger(__name__)


# =============================================================================
# TYPES
# =============================================================================

ModelName = Literal["crnn", "trocr", "florence", "easyocr", "cascade", "all"]


@dataclass
class CompareResult:
    """Résultat de la comparaison des modèles."""
    
    results: Dict[str, SolverResult]
    winner: Optional[str]
    true_text: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "results": {
                name: result.to_dict() for name, result in self.results.items()
            },
            "winner": self.winner,
            "true_text": self.true_text,
        }


@dataclass
class BenchmarkResult:
    """Résultat d'un benchmark."""
    
    model: str
    total_samples: int
    correct: int
    accuracy: float
    avg_time_ms: float
    details: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "model": self.model,
            "total_samples": self.total_samples,
            "correct": self.correct,
            "accuracy": f"{self.accuracy:.2f}%",
            "avg_time_ms": round(self.avg_time_ms, 2),
            "details": self.details,
        }


# =============================================================================
# EASYOCR SOLVER (Fallback simple)
# =============================================================================

class EasyOCRSolver(BaseSolver):
    """
    Solver utilisant EasyOCR comme fallback simple.
    
    EasyOCR est un OCR généraliste, moins précis que TrOCR
    mais utile comme référence.
    """
    
    CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    def __init__(self, languages: List[str] = None) -> None:
        super().__init__(
            name="easyocr",
            model_type=ModelType.EASYOCR,
            charset=self.CHARSET,
        )
        self.languages = languages or ["en"]
        self._reader = None
    
    @property
    def is_available(self) -> bool:
        try:
            import easyocr
            return True
        except ImportError:
            return False
    
    def load(self) -> None:
        if self._is_loaded:
            return
        
        import easyocr
        
        self._reader = easyocr.Reader(
            self.languages,
            gpu=False,
            verbose=False,
        )
        self._is_loaded = True
    
    def solve(self, image_bytes: bytes) -> SolverResult:
        start_time = time.time()
        
        self.ensure_loaded()
        
        import numpy as np
        
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        
        results = self._reader.readtext(image_np)
        
        if not results:
            return SolverResult(
                text="",
                confidence=0.0,
                model=self.name,
                processing_time_ms=round((time.time() - start_time) * 1000, 2),
            )
        
        # Concaténer tous les textes détectés
        text = "".join([r[1] for r in results])
        text = "".join(c for c in text if c.isalnum())
        
        # Confiance moyenne
        avg_confidence = sum(r[2] for r in results) / len(results)
        
        processing_time = (time.time() - start_time) * 1000
        
        return SolverResult(
            text=text.lower(),
            confidence=round(avg_confidence, 3),
            model=self.name,
            processing_time_ms=round(processing_time, 2),
        )
    
    def unload(self) -> None:
        if self._reader is not None:
            del self._reader
            self._reader = None
        self._is_loaded = False


# =============================================================================
# SOLVER SERVICE
# =============================================================================

class SolverService:
    """
    Service de résolution multi-modèle avec cascade.
    
    Gère plusieurs solvers et fournit une interface unifiée pour:
    - Résolution avec un modèle spécifique
    - Résolution en cascade
    - Comparaison des modèles
    - Benchmark
    
    Attributes:
        solvers: Dictionnaire des solvers disponibles.
        cascade_order: Ordre de priorité pour la cascade.
        confidence_threshold: Seuil de confiance pour la cascade.
    """
    
    # Ordre par défaut de la cascade
    DEFAULT_CASCADE_ORDER = ["crnn", "trocr", "florence"]
    
    # Seuil de confiance pour accepter un résultat
    DEFAULT_CONFIDENCE_THRESHOLD = 0.85
    
    def __init__(
        self,
        preload_models: Optional[List[str]] = None,
        cascade_order: Optional[List[str]] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        """
        Initialise le service de résolution.
        
        Args:
            preload_models: Modèles à charger au démarrage (optionnel).
            cascade_order: Ordre de la cascade (optionnel).
            confidence_threshold: Seuil de confiance pour la cascade.
        """
        self.confidence_threshold = confidence_threshold
        self.cascade_order = cascade_order or self.DEFAULT_CASCADE_ORDER
        
        # Initialiser les solvers
        self._solvers: Dict[str, BaseSolver] = {}
        self._init_solvers()
        
        # Précharger certains modèles
        if preload_models:
            for model_name in preload_models:
                if model_name in self._solvers:
                    try:
                        self._solvers[model_name].load()
                        logger.info(f"Modèle {model_name} préchargé")
                    except Exception as e:
                        logger.warning(f"Impossible de précharger {model_name}: {e}")
    
    def _init_solvers(self) -> None:
        """Initialise tous les solvers disponibles."""
        
        # CRNN (modèle de Hana)
        crnn = CRNNSolver()
        if crnn.is_available:
            self._solvers["crnn"] = crnn
            logger.info("CRNN solver disponible")
        else:
            logger.warning("CRNN solver non disponible (fichiers manquants)")
        
        # TrOCR (pré-entraîné HuggingFace)
        trocr = TrOCRSolver()
        if trocr.is_available:
            self._solvers["trocr"] = trocr
            logger.info("TrOCR solver disponible")
        else:
            logger.warning("TrOCR solver non disponible (transformers manquant)")
        
        # Florence-2 (VLM zero-shot)
        florence = FlorenceSolver()
        if florence.is_available:
            self._solvers["florence"] = florence
            logger.info("Florence-2 solver disponible")
        else:
            logger.warning("Florence-2 solver non disponible (transformers manquant)")
        
        # EasyOCR (fallback)
        easyocr = EasyOCRSolver()
        if easyocr.is_available:
            self._solvers["easyocr"] = easyocr
            logger.info("EasyOCR solver disponible")
        else:
            logger.warning("EasyOCR solver non disponible")
    
    @property
    def available_models(self) -> List[str]:
        """Liste des modèles disponibles."""
        return list(self._solvers.keys())
    
    @property
    def loaded_models(self) -> List[str]:
        """Liste des modèles chargés en mémoire."""
        return [name for name, solver in self._solvers.items() if solver.is_loaded]
    
    def get_solver(self, model: str) -> Optional[BaseSolver]:
        """Récupère un solver par son nom."""
        return self._solvers.get(model)
    
    def solve(
        self,
        image_bytes: bytes,
        model: ModelName = "trocr",
    ) -> SolverResult:
        """
        Résout un CAPTCHA avec le modèle spécifié.
        
        Args:
            image_bytes: Image en bytes (PNG/JPG).
            model: Nom du modèle ou 'cascade'/'all'.
            
        Returns:
            SolverResult avec le texte prédit.
            
        Raises:
            ValueError: Si le modèle n'existe pas.
        """
        # Mode cascade
        if model == "cascade":
            return self.solve_cascade(image_bytes)
        
        # Mode all (tous les modèles)
        if model == "all":
            results = self.compare(image_bytes)
            # Retourner le meilleur résultat
            if results.winner and results.winner in results.results:
                return results.results[results.winner]
            # Fallback sur le premier résultat
            for result in results.results.values():
                return result
        
        # Mode spécifique
        if model not in self._solvers:
            available = ", ".join(self.available_models)
            raise ValueError(f"Modèle '{model}' non disponible. Disponibles: {available}")
        
        try:
            return self._solvers[model].solve(image_bytes)
        except Exception as e:
            logger.error(f"Erreur avec {model}: {e}")
            # Essayer un fallback
            for fallback in ["trocr", "easyocr"]:
                if fallback != model and fallback in self._solvers:
                    try:
                        result = self._solvers[fallback].solve(image_bytes)
                        result.metadata = result.metadata or {}
                        result.metadata["fallback_from"] = model
                        return result
                    except Exception:
                        continue
            raise RuntimeError(f"Tous les modèles ont échoué: {e}")
    
    def solve_cascade(
        self,
        image_bytes: bytes,
        order: Optional[List[str]] = None,
    ) -> SolverResult:
        """
        Résout en utilisant la cascade de modèles.
        
        La cascade essaie les modèles dans l'ordre jusqu'à obtenir
        un résultat avec une confiance suffisante.
        
        Args:
            image_bytes: Image en bytes.
            order: Ordre personnalisé (optionnel).
            
        Returns:
            SolverResult du premier modèle confiant ou du dernier.
        """
        cascade = order or self.cascade_order
        
        last_result = None
        tried_models = []
        
        for model_name in cascade:
            if model_name not in self._solvers:
                continue
            
            try:
                result = self._solvers[model_name].solve(image_bytes)
                tried_models.append(model_name)
                last_result = result
                
                # Vérifier la confiance
                if result.confidence is not None and result.confidence >= self.confidence_threshold:
                    logger.info(f"Cascade: {model_name} accepté (conf={result.confidence})")
                    result.metadata = result.metadata or {}
                    result.metadata["cascade_tried"] = tried_models
                    return result
                
                logger.info(f"Cascade: {model_name} rejeté (conf={result.confidence})")
                
            except Exception as e:
                logger.warning(f"Cascade: {model_name} a échoué: {e}")
                tried_models.append(f"{model_name} (erreur)")
        
        # Retourner le dernier résultat
        if last_result:
            last_result.metadata = last_result.metadata or {}
            last_result.metadata["cascade_tried"] = tried_models
            last_result.metadata["cascade_fallback"] = True
            return last_result
        
        raise RuntimeError("Aucun modèle disponible dans la cascade")
    
    def compare(
        self,
        image_bytes: bytes,
        true_text: Optional[str] = None,
        models: Optional[List[str]] = None,
    ) -> CompareResult:
        """
        Compare tous les modèles sur la même image.
        
        Args:
            image_bytes: Image en bytes.
            true_text: Texte réel pour calculer l'accuracy (optionnel).
            models: Liste des modèles à comparer (optionnel, tous par défaut).
            
        Returns:
            CompareResult avec les résultats de chaque modèle.
        """
        models_to_test = models or self.available_models
        results: Dict[str, SolverResult] = {}
        
        for model_name in models_to_test:
            if model_name not in self._solvers:
                continue
            
            try:
                result = self._solvers[model_name].solve(image_bytes)
                
                # Ajouter l'info de correction si true_text fourni
                if true_text:
                    result.metadata = result.metadata or {}
                    result.metadata["correct"] = result.text.lower() == true_text.lower()
                
                results[model_name] = result
                
            except Exception as e:
                logger.warning(f"Comparaison: {model_name} a échoué: {e}")
        
        # Déterminer le gagnant
        winner = self._determine_winner(results, true_text)
        
        return CompareResult(
            results=results,
            winner=winner,
            true_text=true_text,
        )
    
    def _determine_winner(
        self,
        results: Dict[str, SolverResult],
        true_text: Optional[str],
    ) -> Optional[str]:
        """
        Détermine le meilleur modèle.
        
        Critères:
        1. Si true_text fourni: le plus rapide parmi les corrects
        2. Sinon: le plus confiant, puis le plus rapide
        """
        if not results:
            return None
        
        valid_results = [(name, r) for name, r in results.items()]
        
        if true_text:
            # Filtrer les corrects
            correct_results = [
                (name, r) for name, r in valid_results
                if r.text.lower() == true_text.lower()
            ]
            
            if correct_results:
                # Le plus rapide parmi les corrects
                return min(correct_results, key=lambda x: x[1].processing_time_ms)[0]
        
        # Par confiance puis par vitesse
        def sort_key(item):
            name, r = item
            conf = r.confidence if r.confidence is not None else 0
            return (-conf, r.processing_time_ms)
        
        sorted_results = sorted(valid_results, key=sort_key)
        return sorted_results[0][0] if sorted_results else None
    
    def benchmark(
        self,
        n_samples: int = 10,
        noise_level: float = 0.3,
        models: Optional[List[str]] = None,
    ) -> Dict[str, BenchmarkResult]:
        """
        Exécute un benchmark sur des CAPTCHAs générés.
        
        Args:
            n_samples: Nombre de CAPTCHAs à tester.
            noise_level: Niveau de bruit pour la génération.
            models: Modèles à tester (optionnel, tous par défaut).
            
        Returns:
            Dictionnaire {model_name: BenchmarkResult}.
        """
        from app.services.captcha_generator import CaptchaGenerator
        
        generator = CaptchaGenerator()
        models_to_test = models or self.available_models
        
        # Initialiser les résultats
        results: Dict[str, BenchmarkResult] = {
            model: BenchmarkResult(
                model=model,
                total_samples=0,
                correct=0,
                accuracy=0.0,
                avg_time_ms=0.0,
                details=[],
            )
            for model in models_to_test
        }
        
        # Générer et tester
        for i in range(n_samples):
            image, true_text = generator.generate(noise_level=noise_level)
            
            # Convertir en bytes
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            
            # Tester chaque modèle
            for model_name in models_to_test:
                if model_name not in self._solvers:
                    continue
                
                try:
                    result = self._solvers[model_name].solve(image_bytes)
                    is_correct = result.text.lower() == true_text.lower()
                    
                    results[model_name].total_samples += 1
                    results[model_name].avg_time_ms += result.processing_time_ms
                    
                    if is_correct:
                        results[model_name].correct += 1
                    
                    results[model_name].details.append({
                        "true": true_text,
                        "predicted": result.text,
                        "correct": is_correct,
                        "time_ms": result.processing_time_ms,
                        "confidence": result.confidence,
                    })
                    
                except Exception as e:
                    logger.warning(f"Benchmark: {model_name} a échoué sur sample {i}: {e}")
        
        # Calculer les moyennes
        for model_name, result in results.items():
            if result.total_samples > 0:
                result.accuracy = (result.correct / result.total_samples) * 100
                result.avg_time_ms = result.avg_time_ms / result.total_samples
        
        return results
    
    def unload_all(self) -> None:
        """Décharge tous les modèles de la mémoire."""
        for solver in self._solvers.values():
            solver.unload()
        logger.info("Tous les modèles déchargés")
    
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Retourne les informations sur tous les modèles.
        
        Returns:
            Dictionnaire avec les infos de chaque modèle.
        """
        return {
            name: {
                "name": solver.name,
                "type": solver.model_type.value,
                "charset_size": len(solver.charset),
                "is_loaded": solver.is_loaded,
                "is_available": getattr(solver, "is_available", True),
            }
            for name, solver in self._solvers.items()
        }