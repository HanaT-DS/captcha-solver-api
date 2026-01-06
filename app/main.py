"""
CAPTCHA Factory API - Version 2.0
=================================
API de résolution automatique de CAPTCHAs visuels.

Modèles disponibles:
- CRNN : Modèle entraîné par Hana (98% accuracy, 19 chars)
- TrOCR : Modèle pré-entraîné HuggingFace (99% accuracy)
- Florence-2 : VLM zero-shot (universel)
- EasyOCR : OCR généraliste (fallback)

Endpoints:
    GET  /              - Informations de l'API
    GET  /health        - Status de santé
    GET  /models        - Liste des modèles
    POST /generate      - Générer un CAPTCHA
    GET  /generate/random - CAPTCHA aléatoire
    POST /solve         - Résoudre (base64)
    POST /solve/upload  - Résoudre (fichier)
    POST /solve/cascade - Résolution en cascade
    POST /compare       - Comparer les modèles
    GET  /benchmark     - Benchmark complet
    POST /scrape        - Webscraping avec bypass

M2 MoSEF - Université Paris 1 Panthéon-Sorbonne
"""

import io
import base64
import time
import logging
from typing import Optional, List, Literal
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.services.captcha_generator import CaptchaGenerator, CharsetType
from app.services.solver_service import SolverService


# =============================================================================
# CONFIGURATION LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# SERVICES GLOBAUX
# =============================================================================

generator = CaptchaGenerator()
solver = SolverService(preload_models=["trocr"])


# =============================================================================
# LIFECYCLE
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application."""
    # Startup
    logger.info("🚀 CAPTCHA Factory API démarré")
    logger.info(f"📦 Modèles disponibles: {solver.available_models}")
    yield
    # Shutdown
    solver.unload_all()
    logger.info("👋 CAPTCHA Factory API arrêté")


# =============================================================================
# SCHEMAS PYDANTIC
# =============================================================================

class GenerateRequest(BaseModel):
    """Requête de génération de CAPTCHA."""
    length: int = Field(default=5, ge=1, le=10, description="Nombre de caractères")
    width: int = Field(default=200, ge=100, le=400, description="Largeur en pixels")
    height: int = Field(default=60, ge=40, le=150, description="Hauteur en pixels")
    noise_level: float = Field(default=0.3, ge=0, le=1, description="Niveau de bruit (0-1)")
    charset: Optional[str] = Field(None, description="Charset personnalisé")


class GenerateResponse(BaseModel):
    """Réponse de génération."""
    image_base64: str
    true_text: str
    width: int
    height: int
    charset_used: str


class SolveRequest(BaseModel):
    """Requête de résolution."""
    image_base64: str = Field(..., description="Image encodée en base64")
    model: str = Field(default="trocr", description="Modèle: 'crnn', 'trocr', 'florence', 'cascade'")


class SolveResponse(BaseModel):
    """Réponse de résolution."""
    predicted_text: str
    model_used: str
    confidence: Optional[float] = None
    processing_time_ms: float
    metadata: Optional[dict] = None


class CompareRequest(BaseModel):
    """Requête de comparaison."""
    image_base64: str = Field(..., description="Image en base64")
    true_text: Optional[str] = Field(None, description="Texte réel pour vérification")
    models: Optional[List[str]] = Field(None, description="Modèles à comparer")


class CompareResponse(BaseModel):
    """Réponse de comparaison."""
    results: dict
    winner: Optional[str]
    true_text: Optional[str] = None


class ScrapeRequest(BaseModel):
    """Requête de scraping."""
    url: str = Field(..., description="URL à scraper")
    captcha_selector: Optional[str] = Field(None, description="Sélecteur CSS du CAPTCHA")
    input_selector: Optional[str] = Field(None, description="Sélecteur CSS de l'input")
    submit_selector: Optional[str] = Field(None, description="Sélecteur CSS du bouton submit")
    data_selector: Optional[str] = Field(None, description="Sélecteur CSS des données")
    model: str = Field(default="trocr", description="Modèle de résolution")


class ScrapeResponse(BaseModel):
    """Réponse de scraping."""
    success: bool
    url: str
    captcha_found: bool
    captcha_solved: Optional[str] = None
    captcha_confidence: Optional[float] = None
    page_title: Optional[str] = None
    data: Optional[List[str]] = None
    message: str


class ModelInfo(BaseModel):
    """Informations sur un modèle."""
    name: str
    type: str
    charset_size: int
    is_loaded: bool
    is_available: bool


# =============================================================================
# APPLICATION FASTAPI
# =============================================================================

app = FastAPI(
    title="CAPTCHA Factory API",
    description="""
## 🔓 API de résolution automatique de CAPTCHAs visuels

### Fonctionnalités

* **Generate** : Créer des CAPTCHAs personnalisables
* **Solve** : Résoudre des CAPTCHAs avec différents modèles
* **Cascade** : Résolution intelligente avec fallback automatique
* **Compare** : Comparer les performances des modèles
* **Benchmark** : Tester les modèles sur des données générées
* **Scrape** : Webscraping avec bypass automatique

### Modèles disponibles

| Modèle | Accuracy | Charset | Vitesse |
|--------|----------|---------|---------|
| **CRNN** | 98% | 19 chars | ⚡ Rapide |
| **TrOCR** | 99% | Complet | 🔄 Moyen |
| **Florence-2** | ~85% | Universel | 🐢 Lent |
| **EasyOCR** | ~60% | Complet | 🔄 Moyen |

### Cascade recommandée

Pour les meilleurs résultats, utilisez `model: "cascade"` qui essaiera:
1. **CRNN** (si charset compatible) → Rapide
2. **TrOCR** → Très précis
3. **Florence-2** → Fallback universel

---
**M2 MoSEF - Université Paris 1 Panthéon-Sorbonne**
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# GESTIONNAIRE D'ERREURS
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Gère toutes les exceptions non capturées."""
    logger.error(f"Erreur non gérée: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "type": type(exc).__name__,
        }
    )


# =============================================================================
# ENDPOINTS - ROOT & HEALTH
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Point d'entrée de l'API."""
    return {
        "message": "🔓 CAPTCHA Factory API",
        "version": "2.0.0",
        "description": "API de résolution automatique de CAPTCHAs visuels",
        "models": solver.available_models,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "models": "/models",
            "generate": "/generate",
            "solve": "/solve",
            "cascade": "/solve/cascade",
            "compare": "/compare",
            "benchmark": "/benchmark",
            "scrape": "/scrape",
        },
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Vérifie l'état de santé de l'API."""
    return {
        "status": "healthy",
        "models_available": solver.available_models,
        "models_loaded": solver.loaded_models,
    }


@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models():
    """Liste tous les modèles avec leurs informations."""
    info = solver.get_model_info()
    return [ModelInfo(**data) for data in info.values()]


# =============================================================================
# ENDPOINTS - GÉNÉRATION
# =============================================================================

@app.post("/generate", response_model=GenerateResponse, tags=["Generate"])
async def generate_captcha(request: GenerateRequest):
    """
    Génère un nouveau CAPTCHA.
    
    - **length**: Nombre de caractères (1-10)
    - **width**: Largeur de l'image en pixels
    - **height**: Hauteur de l'image en pixels
    - **noise_level**: Niveau de bruit (0-1)
    - **charset**: Charset personnalisé (optionnel)
    """
    try:
        image, text = generator.generate(
            length=request.length,
            width=request.width,
            height=request.height,
            noise_level=request.noise_level,
            charset=request.charset,
        )
        
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return GenerateResponse(
            image_base64=image_base64,
            true_text=text,
            width=request.width,
            height=request.height,
            charset_used=request.charset or generator.charset,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de génération: {str(e)}")


@app.get("/generate/random", response_model=GenerateResponse, tags=["Generate"])
async def generate_random_captcha():
    """Génère un CAPTCHA aléatoire avec les paramètres par défaut."""
    try:
        image, text = generator.generate()
        
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return GenerateResponse(
            image_base64=image_base64,
            true_text=text,
            width=200,
            height=60,
            charset_used=generator.charset,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de génération: {str(e)}")


# =============================================================================
# ENDPOINTS - RÉSOLUTION
# =============================================================================

@app.post("/solve", response_model=SolveResponse, tags=["Solve"])
async def solve_captcha(request: SolveRequest):
    """
    Résout un CAPTCHA à partir d'une image en base64.
    
    - **image_base64**: Image encodée en base64
    - **model**: Modèle à utiliser ('crnn', 'trocr', 'florence', 'easyocr', 'cascade')
    """
    try:
        # Décoder l'image
        try:
            image_bytes = base64.b64decode(request.image_base64)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Image base64 invalide."
            )
        
        # Résoudre
        result = solver.solve(image_bytes, model=request.model)
        
        return SolveResponse(
            predicted_text=result.text,
            model_used=result.model,
            confidence=result.confidence,
            processing_time_ms=result.processing_time_ms,
            metadata=result.metadata,
        )
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de résolution: {str(e)}")


@app.post("/solve/upload", response_model=SolveResponse, tags=["Solve"])
async def solve_uploaded_captcha(
    file: UploadFile = File(..., description="Fichier image PNG ou JPG"),
    model: str = Form(default="trocr", description="Modèle de résolution"),
):
    """
    Résout un CAPTCHA à partir d'un fichier image uploadé.
    
    **Important**: Uploadez une image (PNG, JPG), pas un fichier JSON.
    """
    try:
        # Vérifier le type
        if file.content_type not in ["image/png", "image/jpeg", "image/jpg", "image/webp"]:
            raise HTTPException(
                status_code=400,
                detail=f"Type non supporté: {file.content_type}. Utilisez PNG ou JPG."
            )
        
        image_bytes = await file.read()
        result = solver.solve(image_bytes, model=model)
        
        return SolveResponse(
            predicted_text=result.text,
            model_used=result.model,
            confidence=result.confidence,
            processing_time_ms=result.processing_time_ms,
            metadata=result.metadata,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de résolution: {str(e)}")


@app.post("/solve/cascade", response_model=SolveResponse, tags=["Solve"])
async def solve_cascade(request: SolveRequest):
    """
    Résout un CAPTCHA en utilisant la cascade de modèles.
    
    La cascade essaie les modèles dans l'ordre jusqu'à obtenir
    un résultat confiant:
    1. CRNN (si charset compatible)
    2. TrOCR
    3. Florence-2
    """
    try:
        image_bytes = base64.b64decode(request.image_base64)
        result = solver.solve_cascade(image_bytes)
        
        return SolveResponse(
            predicted_text=result.text,
            model_used=result.model,
            confidence=result.confidence,
            processing_time_ms=result.processing_time_ms,
            metadata=result.metadata,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de cascade: {str(e)}")


# =============================================================================
# ENDPOINTS - COMPARAISON
# =============================================================================

@app.post("/compare", response_model=CompareResponse, tags=["Compare"])
async def compare_models(request: CompareRequest):
    """
    Compare tous les modèles sur le même CAPTCHA.
    
    - **image_base64**: Image en base64
    - **true_text**: Texte réel pour calculer l'accuracy (optionnel)
    - **models**: Liste des modèles à comparer (optionnel)
    """
    try:
        image_bytes = base64.b64decode(request.image_base64)
        compare_result = solver.compare(
            image_bytes,
            true_text=request.true_text,
            models=request.models,
        )
        
        # Convertir les résultats
        results = {}
        for name, result in compare_result.results.items():
            results[name] = {
                "prediction": result.text,
                "confidence": result.confidence,
                "time_ms": result.processing_time_ms,
                "correct": result.metadata.get("correct") if result.metadata else None,
            }
        
        return CompareResponse(
            results=results,
            winner=compare_result.winner,
            true_text=compare_result.true_text,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de comparaison: {str(e)}")


@app.get("/benchmark", tags=["Compare"])
async def run_benchmark(
    n_samples: int = Query(default=10, ge=1, le=100, description="Nombre de CAPTCHAs"),
    noise_level: float = Query(default=0.3, ge=0, le=1, description="Niveau de bruit"),
    models: Optional[str] = Query(None, description="Modèles séparés par des virgules"),
):
    """
    Exécute un benchmark sur des CAPTCHAs générés.
    
    - **n_samples**: Nombre de CAPTCHAs à tester (1-100)
    - **noise_level**: Niveau de bruit (0-1)
    - **models**: Liste des modèles (ex: "trocr,crnn")
    """
    try:
        models_list = models.split(",") if models else None
        results = solver.benchmark(
            n_samples=n_samples,
            noise_level=noise_level,
            models=models_list,
        )
        
        return {
            "n_samples": n_samples,
            "noise_level": noise_level,
            "summary": {
                name: result.to_dict() for name, result in results.items()
            },
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de benchmark: {str(e)}")


# =============================================================================
# ENDPOINTS - WEBSCRAPING
# =============================================================================

@app.post("/scrape", response_model=ScrapeResponse, tags=["Scrape"])
async def scrape_with_captcha(request: ScrapeRequest):
    """
    Scrape une page web en résolvant automatiquement les CAPTCHAs.
    
    **Note**: Nécessite Playwright (`pip install playwright && playwright install chromium`)
    """
    try:
        from app.services.scraper_service import ScraperService
        
        scraper = ScraperService()
        result = await scraper.scrape_with_captcha(
            url=request.url,
            captcha_selector=request.captcha_selector,
            input_selector=request.input_selector,
            submit_selector=request.submit_selector,
            data_selector=request.data_selector,
            model=request.model,
        )
        await scraper.close()
        
        return ScrapeResponse(**result.to_dict())
    
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Playwright non installé. Exécutez: pip install playwright && playwright install chromium"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de scraping: {str(e)}")


@app.get("/scrape/demo", tags=["Scrape"])
async def scrape_demo():
    """Informations sur le module de webscraping."""
    return {
        "message": "Module de Webscraping avec bypass CAPTCHA",
        "description": "Utilisez POST /scrape pour scraper une page avec résolution automatique",
        "test_sites": [
            "https://2captcha.com/demo/normal",
            "https://captcha.com/demos/features/captcha-demo.aspx",
        ],
        "selecteurs_detectes": [
            "img[src*='captcha']",
            "img[id*='captcha']",
            "#captcha",
            ".captcha",
        ],
        "exemple": {
            "url": "https://example.com/login",
            "captcha_selector": "#captcha-image",
            "input_selector": "#captcha-input",
            "submit_selector": "#submit-btn",
            "model": "trocr",
        },
        "note": "Pour des raisons éthiques, utilisez ce module uniquement sur vos propres sites ou avec autorisation.",
    }


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)