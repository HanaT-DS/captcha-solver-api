#!/usr/bin/env python3
"""
CAPTCHA Factory - Point d'entrée principal.

Ce script permet de lancer les différents composants du projet:
- API FastAPI
- Dashboard Streamlit
- Benchmark
- Tests

Usage:
    python main.py api          # Lance l'API FastAPI
    python main.py dashboard    # Lance le dashboard Streamlit
    python main.py benchmark    # Lance le benchmark
    python main.py test         # Lance les tests
    python main.py download     # Télécharge les modèles
    python main.py --help       # Affiche l'aide

M2 MoSEF - Université Paris 1 Panthéon-Sorbonne
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
APP_DIR = PROJECT_ROOT / "app"

# Paramètres par défaut
DEFAULT_API_HOST = "0.0.0.0"
DEFAULT_API_PORT = 8000
DEFAULT_DASHBOARD_PORT = 8501


# =============================================================================
# COMMANDES
# =============================================================================

def run_api(host: str = DEFAULT_API_HOST, port: int = DEFAULT_API_PORT, reload: bool = True):
    """Lance l'API FastAPI."""
    print("🚀 Lancement de l'API FastAPI...")
    print(f"   URL: http://{host}:{port}")
    print(f"   Docs: http://{host}:{port}/docs")
    print()
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", host,
        "--port", str(port),
    ]
    
    if reload:
        cmd.append("--reload")
    
    subprocess.run(cmd, cwd=PROJECT_ROOT)


def run_dashboard(port: int = DEFAULT_DASHBOARD_PORT):
    """Lance le dashboard Streamlit."""
    print("🎨 Lancement du Dashboard Streamlit...")
    print(f"   URL: http://localhost:{port}")
    print()
    
    dashboard_path = APP_DIR / "dashboard.py"
    
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(port),
        "--server.headless", "true",
    ]
    
    subprocess.run(cmd, cwd=PROJECT_ROOT)


def run_benchmark(n_samples: int = 20, models: str = None):
    """Lance le benchmark."""
    print("📊 Lancement du Benchmark...")
    print()
    
    cmd = [
        sys.executable, "scripts/benchmark.py",
        "--n-samples", str(n_samples),
    ]
    
    if models:
        cmd.extend(["--models", models])
    
    subprocess.run(cmd, cwd=PROJECT_ROOT)


def run_tests(verbose: bool = True):
    """Lance les tests pytest."""
    print("🧪 Lancement des Tests...")
    print()
    
    cmd = [sys.executable, "-m", "pytest", "tests/"]
    
    if verbose:
        cmd.append("-v")
    
    subprocess.run(cmd, cwd=PROJECT_ROOT)


def download_models(models: str = "trocr,florence,easyocr"):
    """Télécharge les modèles pré-entraînés."""
    print("📥 Téléchargement des modèles...")
    print()
    
    cmd = [
        sys.executable, "scripts/download_models.py",
        "--models", models,
    ]
    
    subprocess.run(cmd, cwd=PROJECT_ROOT)


def generate_dataset(n: int = 1000, output: str = "./data/synthetic"):
    """Génère un dataset de CAPTCHAs."""
    print("🎲 Génération du dataset...")
    print()
    
    cmd = [
        sys.executable, "scripts/generate_dataset.py",
        "--n", str(n),
        "--output", output,
    ]
    
    subprocess.run(cmd, cwd=PROJECT_ROOT)


def show_info():
    """Affiche les informations du projet."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                     🔓 CAPTCHA Factory                        ║
║                                                               ║
║  API de résolution automatique de CAPTCHAs visuels            ║
║                                                               ║
║  M2 MoSEF - Université Paris 1 Panthéon-Sorbonne              ║
║  Équipe: Hana (CRNN) & Aymen (API/Intégration)                ║
╚═══════════════════════════════════════════════════════════════╝

📦 Modèles disponibles:
   • CRNN      : Modèle entraîné (98% accuracy, 19 chars)
   • TrOCR     : Pré-entraîné HuggingFace (99% accuracy)
   • Florence-2: VLM zero-shot (universel)
   • EasyOCR   : OCR généraliste (fallback)

🚀 Commandes disponibles:
   python main.py api        - Lance l'API FastAPI
   python main.py dashboard  - Lance le dashboard Streamlit
   python main.py benchmark  - Lance le benchmark
   python main.py test       - Lance les tests
   python main.py download   - Télécharge les modèles
   python main.py generate   - Génère un dataset

📚 Documentation:
   • API Docs  : http://localhost:8000/docs
   • Dashboard : http://localhost:8501
    """)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="CAPTCHA Factory - Résolution automatique de CAPTCHAs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python main.py api                    # Lance l'API
  python main.py api --port 8080        # Lance l'API sur le port 8080
  python main.py dashboard              # Lance le dashboard
  python main.py benchmark --n 50       # Benchmark avec 50 échantillons
  python main.py download --models trocr # Télécharge TrOCR uniquement
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")
    
    # Commande: api
    api_parser = subparsers.add_parser("api", help="Lance l'API FastAPI")
    api_parser.add_argument("--host", default=DEFAULT_API_HOST, help="Host")
    api_parser.add_argument("--port", type=int, default=DEFAULT_API_PORT, help="Port")
    api_parser.add_argument("--no-reload", action="store_true", help="Désactiver le hot-reload")
    
    # Commande: dashboard
    dashboard_parser = subparsers.add_parser("dashboard", help="Lance le dashboard Streamlit")
    dashboard_parser.add_argument("--port", type=int, default=DEFAULT_DASHBOARD_PORT, help="Port")
    
    # Commande: benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Lance le benchmark")
    bench_parser.add_argument("--n", type=int, default=20, help="Nombre d'échantillons")
    bench_parser.add_argument("--models", type=str, help="Modèles (ex: trocr,crnn)")
    
    # Commande: test
    test_parser = subparsers.add_parser("test", help="Lance les tests")
    test_parser.add_argument("--quiet", action="store_true", help="Mode silencieux")
    
    # Commande: download
    download_parser = subparsers.add_parser("download", help="Télécharge les modèles")
    download_parser.add_argument("--models", default="trocr,florence,easyocr", help="Modèles à télécharger")
    
    # Commande: generate
    gen_parser = subparsers.add_parser("generate", help="Génère un dataset")
    gen_parser.add_argument("--n", type=int, default=1000, help="Nombre d'images")
    gen_parser.add_argument("--output", default="./data/synthetic", help="Dossier de sortie")
    
    # Commande: info
    subparsers.add_parser("info", help="Affiche les informations du projet")
    
    args = parser.parse_args()
    
    # Exécuter la commande
    if args.command == "api":
        run_api(host=args.host, port=args.port, reload=not args.no_reload)
    
    elif args.command == "dashboard":
        run_dashboard(port=args.port)
    
    elif args.command == "benchmark":
        run_benchmark(n_samples=args.n, models=args.models)
    
    elif args.command == "test":
        run_tests(verbose=not args.quiet)
    
    elif args.command == "download":
        download_models(models=args.models)
    
    elif args.command == "generate":
        generate_dataset(n=args.n, output=args.output)
    
    elif args.command == "info":
        show_info()
    
    else:
        # Pas de commande = afficher l'aide
        show_info()
        print("\nUtilisez --help pour plus d'options.\n")


if __name__ == "__main__":
    main()