"""
CAPTCHA Scraper Service
=======================
Service de webscraping avec detection et resolution automatique de CAPTCHAs.

Utilise Playwright pour l'automatisation web.

M2 MoSEF - Universite Paris 1 Pantheon-Sorbonne
"""

import asyncio
import base64
from typing import Dict, Any, Optional, List
from pathlib import Path

from PIL import Image
import io


class ScraperService:
    """
    Service de webscraping avec bypass CAPTCHA.
    
    Fonctionnalites:
    - Navigation web avec Playwright
    - Detection automatique de CAPTCHAs
    - Resolution et soumission automatique
    - Extraction de donnees
    """
    
    # Selecteurs CSS communs pour les CAPTCHAs
    CAPTCHA_SELECTORS = [
        "img[src*='captcha']",
        "img[alt*='captcha']",
        "img[id*='captcha']",
        "img[class*='captcha']",
        "#captcha-image",
        "#captcha",
        ".captcha-img",
        ".captcha",
        "img[src*='verify']",
        "img[src*='code']",
    ]
    
    # Selecteurs CSS communs pour les inputs CAPTCHA
    INPUT_SELECTORS = [
        "input[name*='captcha']",
        "input[id*='captcha']",
        "input[placeholder*='captcha']",
        "input[placeholder*='code']",
        "#captcha-input",
        "#captcha",
        ".captcha-input",
    ]
    
    def __init__(self):
        """Initialise le service de scraping."""
        self.browser = None
        self.playwright = None
        self.solver = None
    
    async def _init_browser(self):
        """Initialise le navigateur Playwright."""
        if self.browser is None:
            from playwright.async_api import async_playwright
            
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
    
    async def _init_solver(self):
        """Initialise le service de resolution."""
        if self.solver is None:
            from app.services.solver_service import SolverService
            self.solver = SolverService()
    
    async def scrape_with_captcha(
        self,
        url: str,
        captcha_selector: Optional[str] = None,
        input_selector: Optional[str] = None,
        submit_selector: Optional[str] = None,
        data_selector: Optional[str] = None,
        model: str = "easyocr",
    ) -> Dict[str, Any]:
        """
        Scrape une page web en resolvant automatiquement les CAPTCHAs.
        
        Args:
            url: URL de la page a scraper
            captcha_selector: Selecteur CSS pour l'image CAPTCHA
            input_selector: Selecteur CSS pour le champ de saisie
            submit_selector: Selecteur CSS pour le bouton submit
            data_selector: Selecteur CSS pour les donnees a extraire
            model: Modele de resolution a utiliser
        
        Returns:
            Dictionnaire avec les resultats du scraping
        """
        await self._init_browser()
        await self._init_solver()
        
        result = {
            "success": False,
            "url": url,
            "captcha_found": False,
            "captcha_solved": None,
            "page_title": None,
            "data": None,
            "message": "",
        }
        
        try:
            # Creer un nouveau contexte de navigation
            context = await self.browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            page = await context.new_page()
            
            # Naviguer vers l'URL
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            result["page_title"] = await page.title()
            
            # Detecter un CAPTCHA
            captcha_element = await self._detect_captcha(page, captcha_selector)
            
            if captcha_element:
                result["captcha_found"] = True
                
                # Extraire l'image du CAPTCHA
                image_bytes = await self._extract_captcha_image(captcha_element)
                
                if image_bytes:
                    # Resoudre le CAPTCHA
                    solve_result = self.solver.solve(image_bytes, model=model)
                    result["captcha_solved"] = solve_result["text"]
                    
                    # Soumettre la solution si un input est specifie
                    if input_selector:
                        submitted = await self._submit_captcha(
                            page, 
                            input_selector, 
                            solve_result["text"],
                            submit_selector
                        )
                        
                        if submitted:
                            # Attendre le rechargement
                            await page.wait_for_load_state("domcontentloaded")
                            result["message"] = f"CAPTCHA soumis: {solve_result['text']}"
                        else:
                            result["message"] = "CAPTCHA resolu mais soumission echouee"
                    else:
                        result["message"] = f"CAPTCHA resolu: {solve_result['text']}"
                else:
                    result["message"] = "CAPTCHA detecte mais extraction echouee"
            else:
                result["message"] = "Aucun CAPTCHA detecte"
            
            # Extraire les donnees si un selecteur est specifie
            if data_selector:
                data = await self._extract_data(page, data_selector)
                result["data"] = data
            
            result["success"] = True
            
            await context.close()
            
        except Exception as e:
            result["message"] = f"Erreur: {str(e)}"
        
        return result
    
    async def _detect_captcha(
        self, 
        page, 
        custom_selector: Optional[str] = None
    ):
        """
        Detecte un CAPTCHA sur la page.
        
        Args:
            page: Page Playwright
            custom_selector: Selecteur personnalise
        
        Returns:
            Element CAPTCHA ou None
        """
        selectors = [custom_selector] if custom_selector else self.CAPTCHA_SELECTORS
        
        for selector in selectors:
            if selector:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        return element
                except Exception:
                    continue
        
        return None
    
    async def _extract_captcha_image(self, element) -> Optional[bytes]:
        """
        Extrait les bytes de l'image CAPTCHA.
        
        Args:
            element: Element Playwright
        
        Returns:
            Bytes de l'image ou None
        """
        try:
            # Methode 1: Attribut src avec data URL
            src = await element.get_attribute("src")
            
            if src and src.startswith("data:image"):
                # Image encodee en base64
                base64_data = src.split(",")[1]
                return base64.b64decode(base64_data)
            
            # Methode 2: Screenshot de l'element
            screenshot = await element.screenshot()
            return screenshot
            
        except Exception:
            return None
    
    async def _submit_captcha(
        self,
        page,
        input_selector: str,
        solution: str,
        submit_selector: Optional[str] = None,
    ) -> bool:
        """
        Soumet la solution du CAPTCHA.
        
        Args:
            page: Page Playwright
            input_selector: Selecteur du champ input
            solution: Solution a soumettre
            submit_selector: Selecteur du bouton submit
        
        Returns:
            True si soumis avec succes
        """
        try:
            # Trouver et remplir l'input
            input_element = await page.query_selector(input_selector)
            if not input_element:
                # Essayer les selecteurs par defaut
                for selector in self.INPUT_SELECTORS:
                    input_element = await page.query_selector(selector)
                    if input_element:
                        break
            
            if input_element:
                await input_element.fill(solution)
                
                # Soumettre si un bouton est specifie
                if submit_selector:
                    submit_btn = await page.query_selector(submit_selector)
                    if submit_btn:
                        await submit_btn.click()
                else:
                    # Essayer de soumettre avec Enter
                    await input_element.press("Enter")
                
                return True
            
            return False
            
        except Exception:
            return False
    
    async def _extract_data(
        self, 
        page, 
        selector: str
    ) -> Optional[List[str]]:
        """
        Extrait des donnees de la page.
        
        Args:
            page: Page Playwright
            selector: Selecteur CSS
        
        Returns:
            Liste des textes extraits
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
        output_path: Optional[str] = None
    ) -> bytes:
        """
        Prend une capture d'ecran d'une page.
        
        Args:
            url: URL de la page
            output_path: Chemin de sauvegarde (optionnel)
        
        Returns:
            Bytes de l'image PNG
        """
        await self._init_browser()
        
        context = await self.browser.new_context()
        page = await context.new_page()
        
        await page.goto(url, wait_until="domcontentloaded")
        screenshot = await page.screenshot(full_page=False)
        
        if output_path:
            Path(output_path).write_bytes(screenshot)
        
        await context.close()
        
        return screenshot
    
    async def close(self):
        """Ferme le navigateur."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()


# Fonction utilitaire pour utiliser le scraper de maniere synchrone
def scrape_sync(
    url: str,
    captcha_selector: Optional[str] = None,
    input_selector: Optional[str] = None,
    model: str = "easyocr",
) -> Dict[str, Any]:
    """
    Version synchrone du scraper.
    
    Args:
        url: URL a scraper
        captcha_selector: Selecteur du CAPTCHA
        input_selector: Selecteur de l'input
        model: Modele de resolution
    
    Returns:
        Resultats du scraping
    """
    async def _scrape():
        scraper = ScraperService()
        result = await scraper.scrape_with_captcha(
            url=url,
            captcha_selector=captcha_selector,
            input_selector=input_selector,
            model=model,
        )
        await scraper.close()
        return result
    
    return asyncio.run(_scrape())
