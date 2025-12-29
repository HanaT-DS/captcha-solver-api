"""
API FastAPI pour la résolution de CAPTCHAs visuels.
"""
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# Importation des modules locaux
from app.config import MODEL_PATH, MAPPING_PATH
from app.inference import solve_captcha, load_trained_model
from app.schemas import CaptchaResponse, Health


# --- Gestion du cycle de vie (Lifespan) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le chargement unique du modèle au démarrage pour garantir 
    une API robuste et rapide.
    """
    print("⏳ Chargement des ressources ML...")
    
    # Chargement du mapping pour récupérer num_chars
    metadata = joblib.load(MAPPING_PATH)
    num_to_char = metadata["num_to_char"]
    vocab = metadata["vocab"]
    
    # Stockage dans l'état de l'application
    app.state.model = load_trained_model(str(MODEL_PATH), num_chars=len(vocab))
    app.state.mapping = num_to_char
    
    print(f"Modèle chargé avec {len(vocab)} caractères")
    yield
    print("Arrêt du service...")


# --- Initialisation de l'API ---
app = FastAPI(
    title="MOSEF Captcha Solver API",
    description="API industrielle pour la résolution de CAPTCHAs visuels",
    version="1.0.0",
    lifespan=lifespan
)

# Configuration CORS (indispensable pour les tests et le déploiement)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Endpoint : Santé ---
@app.get("/health", response_model=Health, tags=["System"])
async def health() -> Health:
    """Vérifie si le système et le modèle sont opérationnels."""
    is_ready = hasattr(app.state, "model")
    return Health(status="ok" if is_ready else "error", version="1.0.0")


# --- Endpoint : Prédiction ---
@app.post("/predict", response_model=CaptchaResponse, tags=["OCR"])
async def predict_captcha(file: UploadFile = File(...)) -> CaptchaResponse:
    """
    Point d'entrée principal pour le webscraping robuste.
    Prend une image en entrée et renvoie le texte détecté.
    """
    # Validation du format pour éviter les erreurs de traitement
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Format de fichier non supporté.")

    try:
        # Lecture du flux binaire de l'image
        image_bytes = await file.read()
        
        # Inférence avec le modèle chargé dans l'état global
        prediction = solve_captcha(
            image_bytes, 
            app.state.model, 
            app.state.mapping
        )

        return CaptchaResponse(
            text=prediction,
            filename=file.filename
        )

    except Exception as e:
        # Remonte une erreur propre sans exposer la stacktrace complète
        raise HTTPException(status_code=500, detail=f"Échec de l'analyse : {str(e)}")