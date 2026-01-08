"""
Script pour lancer l'API de resolution de CAPTCHAs.

Usage:
    python run_api.py
    
L'API sera disponible sur http://localhost:8000
Documentation: http://localhost:8000/docs
"""

import uvicorn


def main() -> None:
    """Lance le serveur FastAPI."""
    print("=" * 50)
    print("CAPTCHA SOLVER API")
    print("=" * 50)
    print()
    print("Demarrage du serveur...")
    print()
    print("URL:           http://localhost:8000")
    print("Documentation: http://localhost:8000/docs")
    print("Health check:  http://localhost:8000/health")
    print()
    print("-" * 50)
    print("Appuyez sur Ctrl+C pour arreter le serveur")
    print("-" * 50)
    print()
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
