"""
CAPTCHA Factory API
===================
API de resolution automatique de CAPTCHAs visuels.

M2 MoSEF - Universite Paris 1 Pantheon-Sorbonne

Usage:
    uvicorn app.main:app --reload
    
Endpoints:
    GET  /              - Informations de l'API
    GET  /health        - Status de sante
    POST /generate      - Generer un CAPTCHA
    POST /solve         - Resoudre un CAPTCHA (base64)
    POST /solve/upload  - Resoudre un CAPTCHA (fichier)
    POST /compare       - Comparer les modeles
    POST /scrape        - Webscraping avec bypass CAPTCHA
"""

import io
import base64
import time
import asyncio
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.services.captcha_generator import CaptchaGenerator
from app.services.solver_service import SolverService


# =============================================================================
# Schemas Pydantic
# =============================================================================

class GenerateRequest(BaseModel):
    """Schema pour la requete de generation."""
    length: int = Field(default=5, ge=1, le=10, description="Nombre de caracteres")
    width: int = Field(default=200, ge=100, le=400, description="Largeur en pixels")
    height: int = Field(default=60, ge=40, le=150, description="Hauteur en pixels")
    noise_level: float = Field(default=0.3, ge=0, le=1, description="Niveau de bruit")


class GenerateResponse(BaseModel):
    """Schema pour la reponse de generation."""
    image_base64: str
    true_text: str
    width: int
    height: int


class SolveRequest(BaseModel):
    """Schema pour la requete de resolution."""
    image_base64: str = Field(..., description="Image encodee en base64")
    model: str = Field(default="easyocr", description="Modele: 'easyocr', 'crnn', ou 'all'")


class SolveResponse(BaseModel):
    """Schema pour la reponse de resolution."""
    predicted_text: str
    model_used: str
    confidence: Optional[float] = None
    processing_time_ms: float


class CompareRequest(BaseModel):
    """Schema pour la requete de comparaison."""
    image_base64: str = Field(..., description="Image en base64")
    true_text: Optional[str] = Field(None, description="Texte reel pour verification")


class CompareResponse(BaseModel):
    """Schema pour la reponse de comparaison."""
    results: dict
    winner: str
    true_text: Optional[str] = None


class ScrapeRequest(BaseModel):
    """Schema pour la requete de scraping."""
    url: str = Field(..., description="URL a scraper")
    captcha_selector: Optional[str] = Field(None, description="Selecteur CSS du CAPTCHA")
    input_selector: Optional[str] = Field(None, description="Selecteur CSS de l'input")
    submit_selector: Optional[str] = Field(None, description="Selecteur CSS du bouton submit")
    data_selector: Optional[str] = Field(None, description="Selecteur CSS des donnees")
    model: str = Field(default="easyocr", description="Modele de resolution")


class ScrapeResponse(BaseModel):
    """Schema pour la reponse de scraping."""
    success: bool
    url: str
    captcha_found: bool
    captcha_solved: Optional[str] = None
    page_title: Optional[str] = None
    data: Optional[List[str]] = None
    message: str


# =============================================================================
# Application FastAPI
# =============================================================================

app = FastAPI(
    title="CAPTCHA Factory API",
    description="""
## API de resolution automatique de CAPTCHAs visuels

### Fonctionnalites

* **Generate** : Creer des CAPTCHAs personnalisables
* **Solve** : Resoudre des CAPTCHAs avec differents modeles
* **Compare** : Comparer les performances des modeles
* **Scrape** : Webscraping avec bypass automatique de CAPTCHA

### Modeles disponibles

* **EasyOCR** : Modele pre-entraine, rapide sur CPU
* **CRNN** : Modele personnalise entraine sur captcha_images_v2

### Comment utiliser /solve/upload

1. Cliquez sur "Try it out"
2. Cliquez sur "Choose File"
3. Selectionnez une image PNG ou JPG
4. Cliquez sur "Execute"

**Important**: N'uploadez pas le fichier JSON de /generate, mais une vraie image !

---
M2 MoSEF - Universite Paris 1 Pantheon-Sorbonne
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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
# Services (initialises au demarrage)
# =============================================================================

generator = CaptchaGenerator()
solver = SolverService()


# =============================================================================
# Gestionnaire d'erreurs global
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Gere toutes les exceptions non capturees."""
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "type": type(exc).__name__,
        }
    )


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """
    Point d'entree de l'API.
    
    Retourne les informations de base et les endpoints disponibles.
    """
    return {
        "message": "CAPTCHA Factory API",
        "version": "1.0.0",
        "description": "API de resolution automatique de CAPTCHAs visuels",
        "endpoints": {
            "docs": "GET /docs - Documentation Swagger",
            "generate": "POST /generate - Generer un CAPTCHA",
            "solve": "POST /solve - Resoudre un CAPTCHA (base64)",
            "solve_upload": "POST /solve/upload - Resoudre un CAPTCHA (fichier)",
            "compare": "POST /compare - Comparer les modeles",
            "scrape": "POST /scrape - Webscraping avec bypass CAPTCHA",
        },
        "models": solver.get_available_models(),
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Verifie l'etat de sante de l'API."""
    return {
        "status": "healthy",
        "models_available": solver.get_available_models(),
        "models_loaded": solver.get_loaded_models(),
    }


# -----------------------------------------------------------------------------
# Endpoints de generation
# -----------------------------------------------------------------------------

@app.post("/generate", response_model=GenerateResponse, tags=["Generate"])
async def generate_captcha(request: GenerateRequest):
    """
    Genere un nouveau CAPTCHA.
    
    - **length**: Nombre de caracteres (1-10)
    - **width**: Largeur de l'image en pixels
    - **height**: Hauteur de l'image en pixels
    - **noise_level**: Niveau de bruit (0-1)
    
    Retourne l'image en base64 et le texte reel.
    """
    try:
        image, text = generator.generate(
            length=request.length,
            width=request.width,
            height=request.height,
            noise_level=request.noise_level,
        )
        
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return GenerateResponse(
            image_base64=image_base64,
            true_text=text,
            width=request.width,
            height=request.height,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de generation: {str(e)}")


@app.get("/generate/random", response_model=GenerateResponse, tags=["Generate"])
async def generate_random_captcha():
    """Genere un CAPTCHA aleatoire avec les parametres par defaut."""
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
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de generation: {str(e)}")


# -----------------------------------------------------------------------------
# Endpoints de resolution
# -----------------------------------------------------------------------------

@app.post("/solve", response_model=SolveResponse, tags=["Solve"])
async def solve_captcha(request: SolveRequest):
    """
    Resout un CAPTCHA a partir d'une image en base64.
    
    - **image_base64**: Image encodee en base64 (copiez depuis /generate)
    - **model**: Modele a utiliser ('easyocr', 'crnn', ou 'all')
    
    Retourne le texte predit et les metriques.
    """
    try:
        start_time = time.time()
        
        # Decoder l'image
        try:
            image_bytes = base64.b64decode(request.image_base64)
        except Exception:
            raise HTTPException(
                status_code=400, 
                detail="Image base64 invalide. Assurez-vous de copier uniquement la valeur de image_base64."
            )
        
        # Verifier que c'est une image valide
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Le contenu decode n'est pas une image valide."
            )
        
        # Resoudre
        result = solver.solve(image_bytes, model=request.model)
        
        processing_time = (time.time() - start_time) * 1000
        
        return SolveResponse(
            predicted_text=result["text"],
            model_used=result["model"],
            confidence=result.get("confidence"),
            processing_time_ms=round(processing_time, 2),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de resolution: {str(e)}")


@app.post("/solve/upload", response_model=SolveResponse, tags=["Solve"])
async def solve_uploaded_captcha(
    file: UploadFile = File(..., description="Fichier image PNG ou JPG"),
    model: str = Form(default="easyocr", description="Modele: 'easyocr', 'crnn', ou 'all'")
):
    """
    Resout un CAPTCHA a partir d'un fichier image uploade.
    
    **Important**: Uploadez un fichier image (PNG, JPG), pas un fichier JSON !
    
    1. Cliquez sur "Choose File"
    2. Selectionnez une image CAPTCHA
    3. Choisissez le modele
    4. Cliquez sur "Execute"
    """
    try:
        start_time = time.time()
        
        # Verifier le type de fichier
        if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
            raise HTTPException(
                status_code=400,
                detail=f"Type de fichier non supporte: {file.content_type}. "
                       f"Utilisez PNG ou JPG. "
                       f"Note: N'uploadez pas le fichier JSON de /generate !"
            )
        
        # Lire le fichier
        image_bytes = await file.read()
        
        # Verifier que c'est une image valide
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Le fichier n'est pas une image valide."
            )
        
        # Resoudre
        # Re-lire car verify() consomme le stream
        image_bytes = await file.seek(0)
        image_bytes = await file.read()
        
        result = solver.solve(image_bytes, model=model)
        
        processing_time = (time.time() - start_time) * 1000
        
        return SolveResponse(
            predicted_text=result["text"],
            model_used=result["model"],
            confidence=result.get("confidence"),
            processing_time_ms=round(processing_time, 2),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de resolution: {str(e)}")


# -----------------------------------------------------------------------------
# Endpoints de comparaison
# -----------------------------------------------------------------------------

@app.post("/compare", response_model=CompareResponse, tags=["Compare"])
async def compare_models(request: CompareRequest):
    """
    Compare tous les modeles sur le meme CAPTCHA.
    
    Retourne les predictions de chaque modele avec les temps de traitement.
    Si true_text est fourni, calcule aussi la precision.
    """
    try:
        # Decoder l'image
        try:
            image_bytes = base64.b64decode(request.image_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Image base64 invalide.")
        
        results = {}
        
        # Tester chaque modele
        for model_name in solver.get_available_models():
            try:
                start_time = time.time()
                result = solver.solve(image_bytes, model=model_name)
                elapsed = (time.time() - start_time) * 1000
                
                model_result = {
                    "prediction": result["text"],
                    "time_ms": round(elapsed, 2),
                    "confidence": result.get("confidence"),
                }
                
                if request.true_text:
                    model_result["correct"] = (
                        result["text"].lower() == request.true_text.lower()
                    )
                
                results[model_name] = model_result
                
            except Exception as e:
                results[model_name] = {"error": str(e)}
        
        # Determiner le gagnant
        valid_results = [
            (k, v) for k, v in results.items() 
            if "error" not in v
        ]
        
        if valid_results:
            # Si true_text fourni, le gagnant est le plus rapide parmi les corrects
            if request.true_text:
                correct_results = [
                    (k, v) for k, v in valid_results 
                    if v.get("correct", False)
                ]
                if correct_results:
                    winner = min(correct_results, key=lambda x: x[1]["time_ms"])[0]
                else:
                    winner = min(valid_results, key=lambda x: x[1]["time_ms"])[0]
            else:
                winner = min(valid_results, key=lambda x: x[1]["time_ms"])[0]
        else:
            winner = "none"
        
        return CompareResponse(
            results=results,
            winner=winner,
            true_text=request.true_text,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de comparaison: {str(e)}")


@app.get("/compare/benchmark", tags=["Compare"])
async def run_benchmark(n_samples: int = 10):
    """
    Execute un benchmark en generant et resolvant N CAPTCHAs.
    
    Retourne les statistiques de precision et de vitesse pour chaque modele.
    """
    try:
        results = {
            model: {"correct": 0, "total_time": 0, "predictions": []}
            for model in solver.get_available_models()
        }
        
        for i in range(n_samples):
            # Generer un CAPTCHA
            image, true_text = generator.generate()
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            
            # Tester chaque modele
            for model_name in results.keys():
                try:
                    start = time.time()
                    result = solver.solve(image_bytes, model=model_name)
                    elapsed = time.time() - start
                    
                    predicted = result["text"]
                    is_correct = predicted.lower() == true_text.lower()
                    
                    results[model_name]["total_time"] += elapsed
                    if is_correct:
                        results[model_name]["correct"] += 1
                    
                    results[model_name]["predictions"].append({
                        "true": true_text,
                        "predicted": predicted,
                        "correct": is_correct,
                    })
                except Exception:
                    pass
        
        # Calculer les statistiques
        summary = {}
        for model_name, data in results.items():
            n = len(data["predictions"])
            if n > 0:
                summary[model_name] = {
                    "accuracy": f"{data['correct'] / n * 100:.1f}%",
                    "avg_time_ms": round(data["total_time"] / n * 1000, 2),
                    "correct": data["correct"],
                    "total": n,
                }
        
        return {
            "n_samples": n_samples,
            "summary": summary,
            "details": {k: v["predictions"] for k, v in results.items()},
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de benchmark: {str(e)}")


# -----------------------------------------------------------------------------
# Endpoints de webscraping
# -----------------------------------------------------------------------------

@app.post("/scrape", response_model=ScrapeResponse, tags=["Scrape"])
async def scrape_with_captcha(request: ScrapeRequest):
    """
    Scrape une page web en resolvant automatiquement les CAPTCHAs.
    
    - **url**: URL de la page a scraper
    - **captcha_selector**: Selecteur CSS de l'image CAPTCHA (optionnel)
    - **input_selector**: Selecteur CSS du champ de saisie (optionnel)
    - **submit_selector**: Selecteur CSS du bouton submit (optionnel)
    - **data_selector**: Selecteur CSS des donnees a extraire (optionnel)
    - **model**: Modele de resolution ('easyocr' ou 'crnn')
    
    Note: Necessite Playwright installe (pip install playwright && playwright install chromium)
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
        
        return ScrapeResponse(**result)
    
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Playwright non installe. Executez: uv add playwright && playwright install chromium"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de scraping: {str(e)}")


@app.get("/scrape/demo", tags=["Scrape"])
async def scrape_demo():
    """
    Informations sur le module de webscraping.
    """
    return {
        "message": "Module de Webscraping avec bypass CAPTCHA",
        "description": "Utilisez POST /scrape pour scraper une page avec resolution automatique de CAPTCHA",
        "selecteurs_captcha_detectes": [
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
        },
        "note": "Pour des raisons ethiques, utilisez ce module uniquement sur vos propres sites ou avec autorisation.",
    }


# =============================================================================
# Point d'entree
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
