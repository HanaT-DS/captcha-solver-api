#!/usr/bin/env python3
"""
Download Models Script - Télécharge les modèles HuggingFace.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --models trocr,florence,easyocr

M2 MoSEF - Université Paris 1 Panthéon-Sorbonne
"""

import argparse
import sys
import warnings
import os
from pathlib import Path

# Ignorer les warnings non critiques
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, str(Path(__file__).parent.parent))


def download_trocr(cache_dir: str = None) -> bool:
    """Télécharge le modèle TrOCR."""
    print("📥 Téléchargement de TrOCR...")
    
    try:
        from transformers import AutoProcessor, VisionEncoderDecoderModel
        
        model_name = "DunnBC22/trocr-base-printed_captcha_ocr"
        
        print("   Téléchargement du processor...")
        # NE PAS utiliser use_fast=True pour ce modèle
        processor = AutoProcessor.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
        )
        
        print("   Téléchargement du modèle (peut prendre du temps)...")
        model = VisionEncoderDecoderModel.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
        )
        
        # Test de validation
        print("   Vérification du modèle...")
        if processor is None or model is None:
            raise RuntimeError("Le modèle ou le processor est None")
        
        # Vérifier que le modèle a les bonnes méthodes
        if not hasattr(model, 'generate'):
            raise RuntimeError("Le modèle n'a pas de méthode generate()")
        
        print("✅ TrOCR téléchargé avec succès!")
        return True
        
    except Exception as e:
        import traceback
        print(f"❌ Erreur TrOCR: {type(e).__name__}: {e}")
        print("\n📋 Détails de l'erreur:")
        traceback.print_exc()
        print("\n💡 Solutions possibles:")
        print("   1. pip install --upgrade transformers accelerate sentencepiece")
        print("   2. pip cache purge && pip install transformers --force-reinstall")
        return False


def download_florence(cache_dir: str = None) -> bool:
    """Télécharge le modèle Florence-2."""
    print("📥 Téléchargement de Florence-2...")
    print("   ⚠️ Florence-2 est optionnel (~500 MB).")
    
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        import torch
        
        model_name = "microsoft/Florence-2-base"
        
        print("   Téléchargement du processor...")
        processor = AutoProcessor.from_pretrained(
            model_name, 
            cache_dir=cache_dir, 
            trust_remote_code=True,
        )
        
        print("   Téléchargement du modèle (peut prendre du temps)...")
        # Utiliser les nouveaux paramètres compatibles
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir=cache_dir, 
            trust_remote_code=True,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        
        if processor is None or model is None:
            raise RuntimeError("Le modèle ou le processor est None")
        
        print("✅ Florence-2 téléchargé avec succès!")
        return True
        
    except Exception as e:
        import traceback
        print(f"❌ Erreur Florence-2: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("\n💡 Florence-2 est optionnel. TrOCR + CRNN suffisent.")
        return False


def download_easyocr() -> bool:
    """Initialise EasyOCR."""
    print("📥 Initialisation d'EasyOCR...")
    try:
        import easyocr
        
        print("   Téléchargement des modèles EasyOCR...")
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        
        if reader is None:
            raise RuntimeError("EasyOCR Reader est None")
        
        print("✅ EasyOCR initialisé avec succès!")
        return True
        
    except ImportError:
        print("❌ EasyOCR non installé.")
        print("   pip install easyocr")
        return False
    except Exception as e:
        print(f"❌ Erreur EasyOCR: {type(e).__name__}: {e}")
        return False


def check_dependencies():
    """Vérifie les dépendances critiques."""
    print("🔍 Vérification des dépendances...")
    
    issues = []
    
    # Vérifier transformers
    try:
        import transformers
        version = transformers.__version__
        print(f"   ✓ transformers {version}")
        
        major, minor = map(int, version.split(".")[:2])
        if major < 4 or (major == 4 and minor < 40):
            issues.append("transformers >= 4.40.0 recommandé")
    except ImportError:
        issues.append("transformers non installé")
    
    # Vérifier torch
    try:
        import torch
        print(f"   ✓ torch {torch.__version__}")
    except ImportError:
        issues.append("torch non installé")
    
    # Vérifier PIL
    try:
        from PIL import Image
        import PIL
        print(f"   ✓ pillow {PIL.__version__}")
    except ImportError:
        issues.append("pillow non installé")
    
    # Vérifier sentencepiece (nécessaire pour TrOCR)
    try:
        import sentencepiece
        print(f"   ✓ sentencepiece installé")
    except ImportError:
        issues.append("sentencepiece non installé (nécessaire pour TrOCR)")
    
    if issues:
        print("\n⚠️ Problèmes détectés:")
        for issue in issues:
            print(f"   - {issue}")
        print("\n💡 Exécutez:")
        print("   pip install --upgrade transformers accelerate torch pillow sentencepiece")
        return False
    
    print("   ✓ Toutes les dépendances OK\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Télécharge les modèles pré-entraînés")
    parser.add_argument("--models", type=str, default="trocr,easyocr",
                       help="Modèles à télécharger (trocr,florence,easyocr)")
    parser.add_argument("--cache-dir", type=str, default=None,
                       help="Répertoire de cache")
    parser.add_argument("--skip-check", action="store_true",
                       help="Ignorer la vérification des dépendances")
    args = parser.parse_args()
    
    print("=" * 55)
    print("🔓 CAPTCHA Factory - Téléchargement des Modèles")
    print("=" * 55)
    print()
    
    # Vérifier les dépendances
    if not args.skip_check:
        deps_ok = check_dependencies()
        if not deps_ok:
            print("Continuez avec --skip-check pour ignorer.\n")
    
    models = [m.strip().lower() for m in args.models.split(",")]
    
    print(f"📦 Modèles sélectionnés: {', '.join(models)}\n")
    
    results = {}
    
    if "trocr" in models:
        results["trocr"] = download_trocr(args.cache_dir)
        print()
    
    if "florence" in models:
        results["florence"] = download_florence(args.cache_dir)
        print()
    
    if "easyocr" in models:
        results["easyocr"] = download_easyocr()
        print()
    
    # Résumé
    print("=" * 55)
    print("📊 RÉSUMÉ")
    print("=" * 55)
    
    for model, success in results.items():
        status = "✅ OK" if success else "❌ ÉCHEC"
        print(f"   {model.upper():12} : {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print()
    if success_count == total_count:
        print("🎉 Tous les modèles sont prêts!")
    elif success_count > 0:
        print(f"⚠️ {success_count}/{total_count} modèles installés.")
        if "trocr" in results and not results["trocr"]:
            print("\n🔧 Pour corriger TrOCR, essayez:")
            print("   pip install --upgrade transformers accelerate sentencepiece")
            print("   pip cache purge")
            print("   python scripts/download_models.py --models trocr")
    else:
        print("❌ Aucun modèle n'a pu être téléchargé.")
    
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())