"""
Scraper avec Playwright pour automatiser la resolution de CAPTCHAs.

Ce module fournit une classe qui utilise Playwright en mode synchrone
pour naviguer vers un site, capturer le CAPTCHA, l'envoyer a l'API
et soumettre la solution.
"""

import base64
import logging
import time
from typing import Optional

import requests
from playwright.sync_api import sync_playwright, Page, Browser, Playwright


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CaptchaScraper:
    """
    Scraper pour naviguer et resoudre automatiquement les CAPTCHAs.
    
    Utilise Playwright en mode synchrone avec navigateur visible
    pour permettre de voir toutes les etapes.
    
    Attributes:
        api_url: URL de l'API de resolution.
        headless: Mode headless (False = visible).
        slow_mo: Delai entre les actions en ms.
    """
    
    # Configuration du site 2captcha demo
    SITE_URL = "https://2captcha.com/demo/normal"
    
    # Selecteurs CSS pour les elements de la page
    CAPTCHA_SELECTOR = ".captcha_image img, img.captcha_img, img[class*='captcha']"
    INPUT_SELECTOR = "input#simple-captcha-field, input[name*='captcha'], input[data-testid*='captcha']"
    SUBMIT_SELECTOR = "button#btn-verify, button[type='submit']"
    
    # Texte indiquant le succes
    SUCCESS_TEXT = "Captcha is passed successfully"
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        headless: bool = False,
        slow_mo: int = 100,
    ) -> None:
        """
        Initialise le scraper.
        
        Args:
            api_url: URL de l'API de resolution (defaut: localhost:8000).
            headless: Si False, le navigateur est visible (defaut: False).
            slow_mo: Delai en ms entre les actions pour voir les etapes.
        """
        self._api_url = api_url.rstrip("/")
        self._headless = headless
        self._slow_mo = slow_mo
        
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None
        
        logger.info(f"CaptchaScraper initialise - API: {self._api_url}")
    
    def run(self) -> bool:
        """
        Execute le workflow complet de resolution.
        
        Etapes:
            1. Ouvrir le navigateur et naviguer vers le site
            2. Detecter et capturer le CAPTCHA
            3. Envoyer a l'API pour resolution
            4. Saisir la solution et soumettre
            5. Verifier le resultat
        
        Returns:
            True si le CAPTCHA a ete resolu avec succes, False sinon.
        """
        logger.info("=" * 50)
        logger.info("DEMARRAGE DU SCRAPER")
        logger.info("=" * 50)
        
        success = False
        
        with sync_playwright() as playwright:
            self._playwright = playwright
            
            try:
                # Lancer le navigateur
                self._launch_browser()
                
                # Etape 1: Navigation
                if not self._navigate_to_site():
                    return False
                
                # Etape 2: Capture du CAPTCHA
                captcha_bytes = self._extract_captcha()
                if captcha_bytes is None:
                    logger.error("Impossible de capturer le CAPTCHA")
                    return False
                
                # Etape 3: Resolution via API
                solution = self._solve_via_api(captcha_bytes)
                if solution is None:
                    logger.error("Impossible de resoudre le CAPTCHA")
                    return False
                
                # Etape 4: Soumission
                self._submit_solution(solution)
                
                # Etape 5: Verification
                time.sleep(2)
                success = self._check_success()
                
                # Pause pour voir le resultat
                logger.info("Pause de 5 secondes pour voir le resultat...")
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Erreur durant l'execution: {e}")
                success = False
                
            finally:
                self._close_browser()
        
        return success
    
    def _launch_browser(self) -> None:
        """
        Lance le navigateur Chromium.
        
        Le navigateur est configure en mode visible avec un delai
        entre les actions pour pouvoir observer le processus.
        """
        logger.info("Lancement du navigateur...")
        
        self._browser = self._playwright.chromium.launch(
            headless=self._headless,
            slow_mo=self._slow_mo,
        )
        
        self._page = self._browser.new_page()
        
        # Configurer le viewport
        self._page.set_viewport_size({"width": 1280, "height": 720})
        
        logger.info("Navigateur lance")
    
    def _close_browser(self) -> None:
        """Ferme le navigateur proprement."""
        if self._browser is not None:
            logger.info("Fermeture du navigateur...")
            self._browser.close()
            self._browser = None
            self._page = None
    
    def _navigate_to_site(self) -> bool:
        """
        Navigue vers le site cible.
        
        Returns:
            True si la navigation a reussi.
        """
        logger.info(f"Navigation vers: {self.SITE_URL}")
        
        try:
            self._page.goto(self.SITE_URL, wait_until="networkidle", timeout=30000)
            
            # Attendre que la page soit completement chargee
            time.sleep(3)
            
            logger.info("Page chargee avec succes")
            return True
            
        except Exception as e:
            logger.error(f"Erreur de navigation: {e}")
            return False
    
    def _extract_captcha(self) -> Optional[bytes]:
        """
        Extrait l'image du CAPTCHA de la page.
        
        Fait un screenshot uniquement de l'element CAPTCHA,
        pas de toute la page.
        
        Returns:
            Bytes de l'image PNG ou None si echec.
        """
        logger.info("Recherche de l'element CAPTCHA...")
        
        try:
            # Essayer plusieurs selecteurs
            selectors = self.CAPTCHA_SELECTOR.split(", ")
            captcha_element = None
            
            for selector in selectors:
                try:
                    self._page.wait_for_selector(
                        selector.strip(),
                        state="visible",
                        timeout=5000
                    )
                    captcha_element = self._page.query_selector(selector.strip())
                    if captcha_element:
                        logger.info(f"CAPTCHA trouve avec: {selector.strip()}")
                        break
                except Exception:
                    continue
            
            if captcha_element is None:
                # Fallback: chercher toute image avec 'captcha' dans src ou class
                logger.info("Tentative de recherche alternative...")
                all_images = self._page.query_selector_all("img")
                
                for img in all_images:
                    src = img.get_attribute("src") or ""
                    cls = img.get_attribute("class") or ""
                    
                    if "captcha" in src.lower() or "captcha" in cls.lower():
                        captcha_element = img
                        logger.info("CAPTCHA trouve par recherche alternative")
                        break
            
            if captcha_element is None:
                logger.error("Aucun element CAPTCHA trouve")
                return None
            
            # Screenshot de l'element
            screenshot_bytes = captcha_element.screenshot()
            
            logger.info(f"CAPTCHA capture: {len(screenshot_bytes)} bytes")
            
            return screenshot_bytes
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction du CAPTCHA: {e}")
            return None
    
    def _solve_via_api(self, image_bytes: bytes) -> Optional[str]:
        """
        Envoie l'image a l'API et recupere la solution.
        
        Args:
            image_bytes: Image du CAPTCHA en bytes.
            
        Returns:
            Texte de la solution ou None en cas d'erreur.
        """
        logger.info("Envoi de l'image a l'API...")
        
        try:
            # Encoder en Base64
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            
            # Appeler l'API
            response = requests.post(
                f"{self._api_url}/solve",
                json={"image_base64": image_base64},
                timeout=60,
            )
            
            response.raise_for_status()
            data = response.json()
            
            if data.get("success"):
                text = data.get("text", "")
                confidence = data.get("confidence", 0)
                logger.info(f"Solution recue: '{text}' (confiance: {confidence})")
                return text
            else:
                error = data.get("error", "Erreur inconnue")
                logger.error(f"API a retourne une erreur: {error}")
                return None
                
        except requests.exceptions.ConnectionError:
            logger.error(f"Impossible de se connecter a l'API: {self._api_url}")
            logger.error("Verifiez que l'API est lancee (python run_api.py)")
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors de l'appel API: {e}")
            return None
    
    def _submit_solution(self, solution: str) -> None:
        """
        Remplit le champ input et soumet le formulaire.
        
        Args:
            solution: Texte a saisir dans le champ.
        """
        logger.info(f"Saisie de la solution: '{solution}'")
        
        try:
            # Trouver le champ input
            selectors = self.INPUT_SELECTOR.split(", ")
            input_element = None
            
            for selector in selectors:
                try:
                    input_element = self._page.query_selector(selector.strip())
                    if input_element:
                        logger.info(f"Input trouve avec: {selector.strip()}")
                        break
                except Exception:
                    continue
            
            if input_element is None:
                # Fallback
                input_element = self._page.query_selector("input[type='text']")
            
            if input_element:
                # Cliquer sur le champ
                input_element.click()
                
                # Effacer le contenu existant
                input_element.fill("")
                
                # Taper le texte lettre par lettre (visible)
                input_element.type(solution, delay=100)
                
                logger.info("Solution saisie")
            else:
                logger.warning("Champ input non trouve")
            
            # Pause avant de soumettre
            time.sleep(1)
            
            # Trouver et cliquer sur le bouton de soumission
            submit_selectors = self.SUBMIT_SELECTOR.split(", ")
            submit_button = None
            
            for selector in submit_selectors:
                try:
                    submit_button = self._page.query_selector(selector.strip())
                    if submit_button:
                        break
                except Exception:
                    continue
            
            if submit_button is None:
                # Fallback
                submit_button = self._page.query_selector("button")
            
            if submit_button:
                submit_button.click()
                logger.info("Formulaire soumis")
            else:
                logger.warning("Bouton de soumission non trouve")
                # Essayer de soumettre avec Enter
                if input_element:
                    input_element.press("Enter")
                    logger.info("Soumission avec touche Enter")
                    
        except Exception as e:
            logger.error(f"Erreur lors de la soumission: {e}")
    
    def _check_success(self) -> bool:
        """
        Verifie si le CAPTCHA a ete valide avec succes.
        
        Returns:
            True si le message de succes est affiche.
        """
        logger.info("Verification du resultat...")
        
        try:
            # Attendre un peu pour que la page se mette a jour
            time.sleep(2)
            
            # Recuperer le contenu de la page
            page_content = self._page.content()
            
            # Verifier si le texte de succes est present
            if self.SUCCESS_TEXT.lower() in page_content.lower():
                logger.info("SUCCES: CAPTCHA valide!")
                return True
            
            # Verifier d'autres indicateurs de succes
            success_indicators = [
                "success",
                "passed",
                "correct",
                "verified"
            ]
            
            for indicator in success_indicators:
                if indicator in page_content.lower():
                    # Verifier que ce n'est pas un message d'erreur
                    if "error" not in page_content.lower():
                        logger.info(f"SUCCES detecte (indicateur: {indicator})")
                        return True
            
            logger.warning("ECHEC: CAPTCHA non valide")
            return False
            
        except Exception as e:
            logger.error(f"Erreur lors de la verification: {e}")
            return False
