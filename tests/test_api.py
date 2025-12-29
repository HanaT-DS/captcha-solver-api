from fastapi.testclient import TestClient
from app.main import app

# Création du client de test
client = TestClient(app)

def test_health_check():
    """
    Vérifie que la route /health répond 200 et que le statut est 'ok'.
    """
    response = client.get("/health")
    
    # Vérifications (assertions)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_real_image():
    """Vérifie la résolution d'un vrai CAPTCHA de ton dossier."""
    file_path = "data/captcha_images_v2/2b827.png" 
    
    with open(file_path, "rb") as f:
        response = client.post(
            "/predict",
            # 'file' doit correspondre au nom dans le décorateur de main.py
            files={"file": ("2b827.png", f, "image/png")}
        )
    
    # On vérifie que le modèle répond correctement
    assert response.status_code == 200
    # On vérifie que la réponse contient bien un champ "text"
    assert "text" in response.json()
    print(f"Résultat du test : {response.json()['text']}")

    # pytest tests/test_api.py