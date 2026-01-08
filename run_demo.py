"""
Script pour lancer la demo de resolution de CAPTCHAs.

Ce script ouvre un navigateur visible et montre toutes les etapes
du processus de resolution automatique d'un CAPTCHA.

Usage:
    1. D'abord, lancer l'API dans un terminal: python run_api.py
    2. Ensuite, lancer ce script: python run_demo.py
"""

import sys

from scraper.browser import CaptchaScraper


def check_api_available(api_url: str) -> bool:
    """
    Verifie si l'API est accessible.
    
    Args:
        api_url: URL de l'API.
        
    Returns:
        True si l'API repond.
    """
    import requests
    
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def main() -> None:
    """Lance la demo visuelle de resolution de CAPTCHA."""
    print("=" * 50)
    print("DEMO CAPTCHA FACTORY")
    print("=" * 50)
    print()
    
    api_url = "http://localhost:8000"
    
    # Verifier que l'API est lancee
    print(f"Verification de l'API ({api_url})...")
    
    if not check_api_available(api_url):
        print()
        print("ERREUR: L'API n'est pas accessible!")
        print()
        print("Veuillez d'abord lancer l'API dans un autre terminal:")
        print("    python run_api.py")
        print()
        sys.exit(1)
    
    print("API disponible!")
    print()
    print("-" * 50)
    print("Le navigateur va s'ouvrir...")
    print("Vous allez voir toutes les etapes du processus:")
    print("  1. Navigation vers le site")
    print("  2. Detection du CAPTCHA")
    print("  3. Capture de l'image")
    print("  4. Envoi a l'API")
    print("  5. Saisie de la solution")
    print("  6. Verification du resultat")
    print("-" * 50)
    print()
    
    # Creer le scraper
    scraper = CaptchaScraper(
        api_url=api_url,
        headless=False,     # Navigateur visible
        slow_mo=150,        # 150ms entre chaque action
    )
    
    # Lancer le workflow
    success = scraper.run()
    
    # Afficher le resultat
    print()
    print("=" * 50)
    
    if success:
        print("RESULTAT: SUCCES")
        print("Le CAPTCHA a ete resolu correctement!")
    else:
        print("RESULTAT: ECHEC")
        print("Le CAPTCHA n'a pas pu etre resolu.")
        print("Cela peut arriver si:")
        print("  - Le texte etait trop deforme")
        print("  - Les selecteurs CSS ont change")
        print("  - L'API a retourne une mauvaise reponse")
    
    print("=" * 50)


if __name__ == "__main__":
    main()
