# CAPTCHA Factory

Projet de resolution automatique de CAPTCHAs avec EasyOCR et Playwright.

## Architecture

```
+-------------------+         +-------------------+
|     SCRAPER       |  HTTP   |       API         |
|   (Playwright)    | ------> |    (FastAPI)      |
|                   |         |    + EasyOCR      |
|  headless=False   | <------ |                   |
|    (visible)      |  JSON   |                   |
+-------------------+         +-------------------+
```

## Structure du Projet

```
captcha_factory/
|
+-- api/
|   +-- __init__.py
|   +-- main.py          # Endpoint FastAPI POST /solve
|   +-- solver.py        # Classe EasyOCR
|
+-- scraper/
|   +-- __init__.py
|   +-- browser.py       # Classe Playwright (visible)
|
+-- run_api.py           # Lance l'API
+-- run_demo.py          # Lance la demo visuelle
+-- requirements.txt
+-- README.md
```

## Installation

### 1. Creer un environnement virtuel (recommande)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Installer les dependances

```bash
pip install -r requirements.txt
```

### 3. Installer Playwright

```bash
playwright install chromium
```

## Utilisation

### Etape 1: Lancer l'API (Terminal 1)

```bash
python run_api.py
```

L'API sera disponible sur:
- URL: http://localhost:8000
- Documentation: http://localhost:8000/docs

### Etape 2: Lancer la demo (Terminal 2)

```bash
python run_demo.py
```

Le navigateur s'ouvrira et vous pourrez observer:
1. La navigation vers 2captcha.com/demo/normal
2. La detection du CAPTCHA
3. La capture de l'image
4. L'envoi a l'API
5. La saisie automatique de la solution
6. La verification du resultat

## API Endpoints

### POST /solve

Resout un CAPTCHA a partir d'une image Base64.

**Requete:**
```json
{
    "image_base64": "iVBORw0KGgo..."
}
```

**Reponse:**
```json
{
    "success": true,
    "text": "W9H5K",
    "confidence": 0.85
}
```

### GET /health

Verifie que l'API fonctionne.

**Reponse:**
```json
{
    "status": "ok",
    "message": "API is running"
}
```

## Configuration

### Modifier le site cible

Dans `scraper/browser.py`, modifiez les constantes:

```python
SITE_URL = "https://2captcha.com/demo/normal"
CAPTCHA_SELECTOR = "..."
INPUT_SELECTOR = "..."
SUBMIT_SELECTOR = "..."
SUCCESS_TEXT = "..."
```

### Mode headless

Pour executer sans afficher le navigateur:

```python
scraper = CaptchaScraper(
    headless=True,  # Invisible
    slow_mo=0,      # Pas de delai
)
```

## Dependances

- fastapi: Framework API
- uvicorn: Serveur ASGI
- playwright: Automatisation navigateur
- easyocr: OCR pour lire les CAPTCHAs
- Pillow: Manipulation d'images
- requests: Appels HTTP
- numpy: Calcul numerique (requis par EasyOCR)

## Troubleshooting

### "L'API n'est pas accessible"

Assurez-vous que l'API est lancee dans un autre terminal:
```bash
python run_api.py
```

### "CAPTCHA non trouve"

Les selecteurs CSS peuvent avoir change. Inspectez la page et mettez a jour les selecteurs dans `browser.py`.

### "EasyOCR est lent au premier lancement"

C'est normal. EasyOCR telecharge les modeles au premier lancement (~100 MB).

## Licence

MIT
