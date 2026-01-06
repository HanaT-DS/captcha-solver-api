"""
CAPTCHA Scraper Service - Webscraping avec bypass automatique.

Utilise Playwright pour l'automatisation web avec:
- Détection automatique de CAPTCHAs
- Résolution via SolverService
- Soumission automatique
- Extraction de données

M2 MoSEF - Université Paris 1 Panthéon-Sorbonne
"""

import asyncio
import base64
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
import io


# Logger
logger = logging.getLogger(__name__)


# =============================================================================
# SÉLECTEURS CSS PAR DÉFAUT
# =============================================================================

DEFAULT_CAPTCHA_SELECTORS = [
    # Par attribut src
    "img[src*='captcha']",
    "img[src*='verify']",
    "img[src*='code']",
    "img[src*='validation']",
    # Par attribut alt
    "img[alt*='captcha']",
    "img[alt*='verification']",
    # Par ID
    "img[id*='captcha']",
    "#captcha-image",
    "#captcha",
    "#captchaImage",
    # Par classe
    "img[class*='captcha']",
    ".captcha-img",
    ".captcha",
    ".captcha-image",
]

DEFAULT_INPUT_SELECTORS = [
    # Par attribut name
    "input[name*='captcha']",
    "input[name*='verification']",
    "input[name*='code']",
    # Par ID
    "input[id*='captcha']",
    "#captcha-input",
    "#captchaInput",
    "#captcha",
    # Par placeholder
    "input[placeholder*='captcha']",
    "input[placeholder*='code']",
    "input[placeholder*='enter']",
    # Par classe
    ".captcha-input",
]


# =============================================================================
# RÉSULTAT DU SCRAPING
# =============================================================================

@dataclass
class ScrapeResult:
    """Résultat d'une opération de scraping."""
    
    success: bool
    url: str
    captcha_found: bool = False
    captcha_solved: Optional[str] = None
    captcha_confidence: Optional[float] = None
    page_title: Optional[str] = None
    data: Optional[List[str]] = None
    screenshot: Optional[bytes] = None
    message: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "success": self.success,
            "url": self.url,
            "captcha_found": self.captcha_found,
            "captcha_solved": self.captcha_solved,
            "captcha_confidence": self.captcha_confidence,
            "page_title": self.page_title,
            "data": self.data,
            "has_screenshot": self.screenshot is not None,
            "message": self.message,
            "error": self.error,
        }


# =============================================================================
# SCRAPER SERVICE
# =============================================================================

class ScraperService:
    """
    Service de webscraping avec bypass CAPTCHA automatique.
    
    Utilise Playwright pour la navigation et SolverService
    pour la résolution des CAPTCHAs.
    
    Attributes:
        captcha_selectors: Sélecteurs CSS pour les images CAPTCHA.
        input_selectors: Sélecteurs CSS pour les champs input.
    """
    
    def __init__(
        self,
        captcha_selectors: Optional[List[str]] = None,
        input_selectors: Optional[List[str]] = None,
        default_model: str = "trocr",
    ) -> None:
        """
        Initialise le service de scraping.
        
        Args:
            captcha_selectors: Sélecteurs personnalisés pour les CAPTCHAs.
            input_selectors: Sélecteurs personnalisés pour les inputs.
            default_model: Modèle de résolution par défaut.
        """
        self.captcha_selectors = captcha_selectors or DEFAULT_CAPTCHA_SELECTORS
        self.input_selectors = input_selectors or DEFAULT_INPUT_SELECTORS
        self.default_model = default_model
        
        # Playwright (initialisé à la demande)
        self._browser = None
        self._playwright = None
        
        # Solver (initialisé à la demande)
        self._solver = None
    
    async def _init_browser(self) -> None:
        """Initialise le navigateur Playwright."""
        if self._browser is not None:
            return
        
        from playwright.async_api import async_playwright
        
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
        )
        
        logger.info("Navigateur Playwright initialisé")
    
    async def _init_solver(self) -> None:
        """Initialise le service de résolution."""
        if self._solver is not None:
            return
        
        from app.services.solver_service import SolverService
        
        self._solver = SolverService()
        logger.info("SolverService initialisé")
    
    async def scrape_with_captcha(
        self,
        url: str,
        captcha_selector: Optional[str] = None,
        input_selector: Optional[str] = None,
        submit_selector: Optional[str] = None,
        data_selector: Optional[str] = None,
        model: Optional[str] = None,
        take_screenshot: bool = False,
        timeout: int = 30000,
    ) -> ScrapeResult:
        """
        Scrape une page web en résolvant automatiquement les CAPTCHAs.
        
        Args:
            url: URL de la page à scraper.
            captcha_selector: Sélecteur CSS pour l'image CAPTCHA.
            input_selector: Sélecteur CSS pour le champ de saisie.
            submit_selector: Sélecteur CSS pour le bouton submit.
            data_selector: Sélecteur CSS pour les données à extraire.
            model: Modèle de résolution ('crnn', 'trocr', 'florence', 'cascade').
            take_screenshot: Si True, capture une screenshot.
            timeout: Timeout en millisecondes.
            
        Returns:
            ScrapeResult avec les résultats du scraping.
        """
        await self._init_browser()
        await self._init_solver()
        
        model = model or self.default_model
        
        result = ScrapeResult(
            success=False,
            url=url,
        )
        
        try:
            # Créer un nouveau contexte de navigation
            context = await self._browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            )
            page = await context.new_page()
            
            # Naviguer vers l'URL
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            result.page_title = await page.title()
            
            # Détecter un CAPTCHA
            captcha_element = await self._detect_captcha(page, captcha_selector)
            
            if captcha_element:
                result.captcha_found = True
                logger.info(f"CAPTCHA détecté sur {url}")
                
                # Extraire l'image du CAPTCHA
                image_bytes = await self._extract_captcha_image(captcha_element)
                
                if image_bytes:
                    # Résoudre le CAPTCHA
                    solve_result = self._solver.solve(image_bytes, model=model)
                    result.captcha_solved = solve_result.text
                    result.captcha_confidence = solve_result.confidence
                    
                    logger.info(f"CAPTCHA résolu: {solve_result.text} (conf={solve_result.confidence})")
                    
                    # Soumettre la solution si un input est spécifié
                    if input_selector or self.input_selectors:
                        submitted = await self._submit_captcha(
                            page,
                            input_selector,
                            solve_result.text,
                            submit_selector,
                        )
                        
                        if submitted:
                            await page.wait_for_load_state("domcontentloaded")
                            result.message = f"CAPTCHA soumis: {solve_result.text}"
                        else:
                            result.message = "CAPTCHA résolu mais soumission échouée"
                    else:
                        result.message = f"CAPTCHA résolu: {solve_result.text}"
                else:
                    result.message = "CAPTCHA détecté mais extraction échouée"
            else:
                result.message = "Aucun CAPTCHA détecté"
            
            # Extraire les données si un sélecteur est spécifié
            if data_selector:
                data = await self._extract_data(page, data_selector)
                result.data = data
            
            # Screenshot si demandé
            if take_screenshot:
                result.screenshot = await page.screenshot(full_page=False)
            
            result.success = True
            
            await context.close()
            
        except Exception as e:
            logger.error(f"Erreur de scraping: {e}")
            result.error = str(e)
            result.message = f"Erreur: {str(e)}"
        
        return result
    
    async def _detect_captcha(
        self,
        page,
        custom_selector: Optional[str] = None,
    ):
        """
        Détecte un CAPTCHA sur la page.
        
        Args:
            page: Page Playwright.
            custom_selector: Sélecteur personnalisé (prioritaire).
            
        Returns:
            Élément CAPTCHA ou None.
        """
        # Essayer le sélecteur personnalisé d'abord
        if custom_selector:
            try:
                element = await page.query_selector(custom_selector)
                if element:
                    return element
            except Exception:
                pass
        
        # Essayer les sélecteurs par défaut
        for selector in self.captcha_selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    logger.debug(f"CAPTCHA trouvé avec: {selector}")
                    return element
            except Exception:
                continue
        
        return None
    
    async def _extract_captcha_image(self, element) -> Optional[bytes]:
        """
        Extrait les bytes de l'image CAPTCHA.
        
        Args:
            element: Élément Playwright.
            
        Returns:
            Bytes de l'image ou None.
        """
        try:
            # Méthode 1: Attribut src avec data URL
            src = await element.get_attribute("src")
            
            if src and src.startswith("data:image"):
                # Image encodée en base64
                base64_data = src.split(",")[1]
                return base64.b64decode(base64_data)
            
            # Méthode 2: Screenshot de l'élément
            screenshot = await element.screenshot()
            return screenshot
            
        except Exception as e:
            logger.error(f"Erreur d'extraction: {e}")
            return None
    
    async def _submit_captcha(
        self,
        page,
        input_selector: Optional[str],
        solution: str,
        submit_selector: Optional[str] = None,
    ) -> bool:
        """
        Soumet la solution du CAPTCHA.
        
        Args:
            page: Page Playwright.
            input_selector: Sélecteur du champ input.
            solution: Solution à soumettre.
            submit_selector: Sélecteur du bouton submit.
            
        Returns:
            True si soumis avec succès.
        """
        try:
            # Trouver l'input
            input_element = None
            
            if input_selector:
                input_element = await page.query_selector(input_selector)
            
            if not input_element:
                for selector in self.input_selectors:
                    input_element = await page.query_selector(selector)
                    if input_element:
                        break
            
            if not input_element:
                logger.warning("Input CAPTCHA non trouvé")
                return False
            
            # Remplir l'input
            await input_element.fill(solution)
            
            # Soumettre
            if submit_selector:
                submit_btn = await page.query_selector(submit_selector)
                if submit_btn:
                    await submit_btn.click()
            else:
                # Essayer avec Enter
                await input_element.press("Enter")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur de soumission: {e}")
            return False
    
    async def _extract_data(
        self,
        page,
        selector: str,
    ) -> Optional[List[str]]:
        """
        Extrait des données de la page.
        
        Args:
            page: Page Playwright.
            selector: Sélecteur CSS.
            
        Returns:
            Liste des textes extraits.
        """
        try:
            elements = await page.query_selector_all(selector)
            data = []
            
            for element in elements:
                text = await element.text_content()
                if text:
                    data.append(text.strip())
            
            return data if data else None
            
        except Exception:
            return None
    
    async def take_screenshot(
        self,
        url: str,
        output_path: Optional[str] = None,
        full_page: bool = False,
    ) -> bytes:
        """
        Prend une capture d'écran d'une page.
        
        Args:
            url: URL de la page.
            output_path: Chemin de sauvegarde (optionnel).
            full_page: Si True, capture la page entière.
            
        Returns:
            Bytes de l'image PNG.
        """
        await self._init_browser()
        
        context = await self._browser.new_context()
        page = await context.new_page()
        
        await page.goto(url, wait_until="domcontentloaded")
        screenshot = await page.screenshot(full_page=full_page)
        
        if output_path:
            Path(output_path).write_bytes(screenshot)
        
        await context.close()
        
        return screenshot
    
    async def close(self) -> None:
        """Ferme le navigateur et libère les ressources."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        
        logger.info("ScraperService fermé")


# =============================================================================
# FONCTION SYNCHRONE
# =============================================================================

def scrape_sync(
    url: str,
    captcha_selector: Optional[str] = None,
    input_selector: Optional[str] = None,
    submit_selector: Optional[str] = None,
    model: str = "trocr",
) -> Dict[str, Any]:
    """
    Version synchrone du scraper.
    
    Args:
        url: URL à scraper.
        captcha_selector: Sélecteur du CAPTCHA.
        input_selector: Sélecteur de l'input.
        submit_selector: Sélecteur du bouton submit.
        model: Modèle de résolution.
        
    Returns:
        Dictionnaire avec les résultats.
    """
    async def _scrape():
        scraper = ScraperService()
        result = await scraper.scrape_with_captcha(
            url=url,
            captcha_selector=captcha_selector,
            input_selector=input_selector,
            submit_selector=submit_selector,
            model=model,
        )
        await scraper.close()
        return result.to_dict()
    
    return asyncio.run(_scrape())