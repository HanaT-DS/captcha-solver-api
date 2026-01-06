# 🔓 CAPTCHA Factory

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**API de résolution automatique de CAPTCHAs visuels**

[Documentation](#-documentation) • [Installation](#-installation) • [Usage](#-usage) • [API](#-api) • [Modèles](#-modèles)

</div>

---

## 📚 Projet Académique

| | |
|---|---|
| **Formation** | M2 MoSEF (Modélisation Statistique, Économique et Financière) |
| **Université** | Paris 1 Panthéon-Sorbonne |
| **Équipe** | Hana (CRNN) & Aymen (API/Intégration) |
| **Année** | 2024-2025 |

---

## ✨ Fonctionnalités

- 🎨 **Génération** de CAPTCHAs personnalisables
- 🔍 **Résolution** avec plusieurs modèles (CRNN, TrOCR, Florence-2)
- 🔄 **Mode Cascade** avec fallback automatique
- ⚖️ **Comparaison** des performances entre modèles
- 📊 **Benchmark** automatisé
- 🌐 **Webscraping** avec bypass CAPTCHA
- 🎯 **Dashboard** interactif Streamlit

---

## 🤖 Modèles Disponibles

| Modèle | Description | Accuracy | Charset | Vitesse |
|--------|-------------|----------|---------|---------|
| **CRNN** | CNN + GRU bidirectionnel (Hana) | 98% | 19 chars | ⚡ ~50ms |
| **TrOCR** | Transformer pré-entraîné | 99.25% | Complet | 🔄 ~200ms |
| **Florence-2** | VLM Microsoft (zero-shot) | ~85% | Universel | 🐢 ~500ms |
| **EasyOCR** | OCR généraliste | ~60% | Complet | 🔄 ~150ms |

### Mode Cascade

Le mode `cascade` essaie les modèles dans l'ordre jusqu'à obtenir une prédiction confiante :

```
CAPTCHA → CRNN (si charset compatible)
            ↓ (confiance < 85%)
         TrOCR (très précis)
            ↓ (confiance < 85%)
         Florence-2 (fallback universel)
```

---

## 🚀 Installation

### Prérequis

- Python 3.12+
- pip ou uv

### Installation rapide

```bash
# Cloner le projet
git clone https://github.com/username/captcha-factory.git
cd captcha-factory

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Télécharger les modèles pré-entraînés
python scripts/download_models.py

# (Optionnel) Installer Playwright pour le webscraping
pip install playwright
playwright install chromium
```

### Avec uv (recommandé)

```bash
# Installer uv
pip install uv

# Synchroniser les dépendances
uv sync

# Télécharger les modèles
uv run python scripts/download_models.py
```

---

## 💻 Usage

### Lancer l'API

```bash
# Via le script principal
python main.py api

# Ou directement avec uvicorn
uvicorn app.main:app --reload

# L'API est disponible sur http://localhost:8000
# Documentation Swagger: http://localhost:8000/docs
```

### Lancer le Dashboard

```bash
python main.py dashboard

# Le dashboard est disponible sur http://localhost:8501
```

### Autres commandes

```bash
# Télécharger les modèles
python main.py download

# Lancer le benchmark
python main.py benchmark --n 50

# Lancer les tests
python main.py test

# Générer un dataset
python main.py generate --n 1000 --output ./data/synthetic

# Afficher l'aide
python main.py --help
```

---

## 🔌 API

### Endpoints principaux

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Informations de l'API |
| GET | `/health` | Status de santé |
| GET | `/models` | Liste des modèles |
| POST | `/generate` | Générer un CAPTCHA |
| GET | `/generate/random` | CAPTCHA aléatoire |
| POST | `/solve` | Résoudre (base64) |
| POST | `/solve/upload` | Résoudre (fichier) |
| POST | `/solve/cascade` | Résolution en cascade |
| POST | `/compare` | Comparer les modèles |
| GET | `/benchmark` | Benchmark complet |
| POST | `/scrape` | Webscraping avec bypass |

### Exemples d'utilisation

#### Python

```python
import requests
import base64

# Générer un CAPTCHA
response = requests.post("http://localhost:8000/generate", json={
    "length": 5,
    "noise_level": 0.3,
})
data = response.json()
image_base64 = data["image_base64"]
true_text = data["true_text"]

# Résoudre le CAPTCHA
response = requests.post("http://localhost:8000/solve", json={
    "image_base64": image_base64,
    "model": "trocr",
})
result = response.json()
print(f"Prédit: {result['predicted_text']}")
print(f"Confiance: {result['confidence']}")
```

#### cURL

```bash
# Générer un CAPTCHA
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"length": 5, "noise_level": 0.3}'

# Résoudre avec un fichier
curl -X POST "http://localhost:8000/solve/upload" \
     -F "file=@captcha.png" \
     -F "model=trocr"
```

---

## 📁 Structure du Projet

```
CAPTCHA_Factory/
├── app/                          # Code source principal
│   ├── main.py                   # API FastAPI
│   ├── dashboard.py              # Dashboard Streamlit
│   ├── models/                   # Définitions des modèles
│   │   ├── base_solver.py        # Classe abstraite
│   │   ├── crnn_model.py         # Solver CRNN
│   │   ├── trocr_solver.py       # Solver TrOCR
│   │   └── florence_solver.py    # Solver Florence-2
│   ├── services/                 # Services métier
│   │   ├── captcha_generator.py  # Générateur
│   │   ├── solver_service.py     # Service multi-modèle
│   │   └── scraper_service.py    # Webscraping
│   └── utils/                    # Utilitaires
├── config/                       # Configuration
├── data/                         # Données et datasets
├── models/                       # Poids des modèles
├── notebooks/                    # Notebooks Jupyter
├── scripts/                      # Scripts utilitaires
├── tests/                        # Tests unitaires
├── main.py                       # Point d'entrée
├── requirements.txt              # Dépendances
└── README.md                     # Ce fichier
```

---

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| `01_captcha_generator.ipynb` | Démonstration du générateur |
| `02_captcha_solver.ipynb` | Test des modèles de résolution |
| `03_webscraping.ipynb` | Webscraping avec bypass |
| `04_model_comparison.ipynb` | Comparaison des modèles |
| `05_benchmark_complete.ipynb` | Benchmark approfondi |

---

## ⚙️ Configuration

Copiez `.env.example` vers `.env` et modifiez selon vos besoins :

```bash
cp .env.example .env
```

Variables principales :

```env
DEVICE=cpu                    # cpu ou cuda
DEFAULT_MODEL=trocr           # Modèle par défaut
CONFIDENCE_THRESHOLD=0.85     # Seuil pour cascade
LOG_LEVEL=INFO                # Niveau de log
```

---

## 🧪 Tests

```bash
# Tous les tests
pytest tests/ -v

# Tests spécifiques
pytest tests/test_generator.py -v
pytest tests/test_solvers.py -v
pytest tests/test_api.py -v

# Avec couverture
pytest tests/ --cov=app --cov-report=html
```

---

## 📊 Benchmark

```bash
# Benchmark rapide (20 échantillons)
python scripts/benchmark.py

# Benchmark complet (100 échantillons)
python scripts/benchmark.py --n-samples 100 --models trocr,crnn

# Sauvegarder les résultats
python scripts/benchmark.py --output results.json
```

---

## ⚠️ Avertissement

Ce projet est à but **éducatif uniquement**. N'utilisez pas ces outils pour contourner des CAPTCHAs sur des sites sans autorisation. Le contournement de CAPTCHAs peut violer les conditions d'utilisation des sites web et potentiellement des lois locales.

---

## 📄 Licence

MIT License - voir [LICENSE](LICENSE)

---

## 🙏 Remerciements

- [HuggingFace](https://huggingface.co/) pour les modèles pré-entraînés
- [DunnBC22](https://huggingface.co/DunnBC22) pour le modèle TrOCR fine-tuné
- [Microsoft](https://huggingface.co/microsoft) pour Florence-2
- [FastAPI](https://fastapi.tiangolo.com/) et [Streamlit](https://streamlit.io/)

---

<div align="center">

**M2 MoSEF - Université Paris 1 Panthéon-Sorbonne**

Hana & Aymen • 2024-2025

</div>
