# Captcha Solver API 🔓

API de résolution automatique de CAPTCHAs visuels - Projet M2 MoSEF 2025-2026

## 📋 Description

Ce projet vise à concevoir un système de webscraping robuste aux CAPTCHAs visuels en utilisant :
- **Webscraping** : Extraction de données web
- **Computer Vision** : Traitement d'images
- **VLM** (Vision Language Models) : Reconnaissance de texte/contenu
- **FastAPI** : API REST pour industrialiser le projet

## 🛠️ Technologies

- Python 3.12+
- FastAPI
- Pydantic
- pytest
- Ruff (linter/formatter)
- uv (gestionnaire de packages)

## 🚀 Installation

### Prérequis

- Python 3.12 ou supérieur
- [uv](https://github.com/astral-sh/uv) (gestionnaire de packages)

### Installer uv
```bash
# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Cloner le projet
```bash
git clone https://github.com/TON_USERNAME/captcha-solver-api.git
cd captcha-solver-api
```

### Créer l'environnement virtuel
```bash
# Créer le venv
uv venv

# Activer le venv
# Linux/Mac
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat

# Windows (Git Bash)
source .venv/Scripts/activate
```

### Installer les dépendances
```bash
# Production + Dev
uv pip install -e ".[dev]"
```

### Configurer l'environnement
```bash
# Copier le fichier de config
cp .env.example .env

# Éditer si nécessaire
```

## 🏃 Lancer le projet

### API
```bash
uvicorn captcha_solver.main:app --reload
```

L'API sera accessible sur :
- http://localhost:8000
- Documentation Swagger : http://localhost:8000/docs
- Documentation ReDoc : http://localhost:8000/redoc

### Jupyter Notebook
```bash
jupyter notebook
```

## 🧪 Tests et qualité de code
```bash
# Lancer les tests
pytest -v

# Tests avec couverture
pytest --cov=src/captcha_solver

# Linter
ruff check src/

# Linter avec correction auto
ruff check src/ --fix

# Formatter
ruff format src/

# Type checking
mypy src/
```

## 📁 Structure du projet
```
captcha-solver-api/
├── .env.example          # Variables d'environnement (exemple)
├── .gitignore            # Fichiers ignorés par Git
├── pyproject.toml        # Configuration du projet
├── ruff.toml             # Configuration Ruff
├── README.md             # Ce fichier
├── data/
│   ├── raw/              # Données brutes (CAPTCHAs)
│   └── processed/        # Données traitées
├── notebooks/            # Jupyter notebooks
│   └── 01_exploration.ipynb
├── src/
│   └── captcha_solver/
│       ├── __init__.py
│       ├── main.py       # Point d'entrée FastAPI
│       ├── api/
│       │   └── routes/   # Endpoints de l'API
│       ├── core/
│       │   └── config.py # Configuration
│       ├── models/       # Modèles Pydantic
│       └── services/     # Logique métier
└── tests/
    ├── __init__.py
    └── test_main.py
```

## 🔄 Workflow Git

### Branches

- `main` : Branche principale (stable)
- `dev/nom` : Branches de développement individuelles

### Contribuer
```bash
# 1. Récupérer les dernières modifications
git checkout main
git pull origin main

# 2. Créer une branche pour ta feature
git checkout -b feature/ma-feature

# 3. Faire tes modifications
# ...

# 4. Vérifier la qualité du code
ruff check src/ --fix
ruff format src/
pytest -v

# 5. Commiter
git add .
git commit -m "feat: description de la feature"

# 6. Pusher
git push origin feature/ma-feature

# 7. Créer une Pull Request sur GitHub
```

### Conventions de commit

- `feat:` nouvelle fonctionnalité
- `fix:` correction de bug
- `docs:` documentation
- `test:` ajout/modification de tests
- `refactor:` refactoring de code

## 📊 Datasets recommandés

| Dataset | Description |
|---------|-------------|
| LCSD Captcha Dataset | ~6000 images, 4 caractères |
| Captcha Object Detection | ~100k images, 650k objets annotés |
| Pixel Digit Captcha | Chiffres uniquement (5 digits) |
| CAPTCHA Characters | 118k+ images de caractères |
| CAPTCHA Image Dataset | 10k images annotées |

## 👥 Équipe

- Membre 1 - @github_username
- Membre 2 - @github_username

## 📝 Livrables

- [ ] Notebook d'exploration et résultats
- [ ] Rapport Overleaf
- [ ] API fonctionnelle
- [ ] Tests unitaires

## 📄 License

Ce projet est réalisé dans le cadre du Master 2 MoSEF - Université Paris 1 Panthéon-Sorbonne.

helloooo
