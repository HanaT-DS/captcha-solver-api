#!/usr/bin/env python3
"""
Generate Dataset Script - Génère un dataset de CAPTCHAs.

Usage:
    python scripts/generate_dataset.py --n 1000 --output ./data/synthetic
    python scripts/generate_dataset.py --n 500 --charset digits --length 4

M2 MoSEF - Université Paris 1 Panthéon-Sorbonne
"""

import argparse
import os
import sys
import json
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_single(args):
    """Génère un seul CAPTCHA (pour multiprocessing)."""
    idx, generator, length, noise_range, output_dir = args
    
    noise = random.uniform(*noise_range)
    image, text = generator.generate(length=length, noise_level=noise)
    
    # Sauvegarder l'image
    filename = f"{text}_{idx:05d}.png"
    filepath = output_dir / filename
    image.save(filepath)
    
    return {"filename": filename, "text": text, "noise": noise}


def main():
    parser = argparse.ArgumentParser(description="Génère un dataset de CAPTCHAs")
    parser.add_argument("--n", type=int, default=1000, help="Nombre d'images")
    parser.add_argument("--output", type=str, default="./data/synthetic", help="Dossier de sortie")
    parser.add_argument("--charset", type=str, default="crnn", choices=["crnn", "digits", "alpha", "full"])
    parser.add_argument("--length", type=int, default=5, help="Longueur du texte")
    parser.add_argument("--noise-min", type=float, default=0.1, help="Bruit minimum")
    parser.add_argument("--noise-max", type=float, default=0.5, help="Bruit maximum")
    parser.add_argument("--width", type=int, default=200, help="Largeur")
    parser.add_argument("--height", type=int, default=60, help="Hauteur")
    parser.add_argument("--workers", type=int, default=4, help="Nombre de workers")
    parser.add_argument("--split", action="store_true", help="Créer train/val/test splits")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔓 CAPTCHA Factory - Génération de Dataset")
    print("=" * 60)
    
    # Charsets
    charsets = {
        "crnn": "23456789bcdefgmnpwxy",
        "digits": "0123456789",
        "alpha": "abcdefghijklmnopqrstuvwxyz",
        "full": "0123456789abcdefghijklmnopqrstuvwxyz",
    }
    charset = charsets[args.charset]
    
    print(f"📊 Configuration:")
    print(f"   Nombre: {args.n}")
    print(f"   Charset: {args.charset} ({len(charset)} chars)")
    print(f"   Longueur: {args.length}")
    print(f"   Bruit: {args.noise_min}-{args.noise_max}")
    print(f"   Dimensions: {args.width}x{args.height}")
    print()
    
    # Créer le dossier de sortie
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Générateur
    from app.services.captcha_generator import CaptchaGenerator
    generator = CaptchaGenerator(charset=charset)
    
    # Générer
    print(f"⏳ Génération de {args.n} CAPTCHAs...")
    
    metadata = []
    noise_range = (args.noise_min, args.noise_max)
    
    for i in tqdm(range(args.n)):
        noise = random.uniform(*noise_range)
        image, text = generator.generate(
            length=args.length,
            width=args.width,
            height=args.height,
            noise_level=noise,
        )
        
        filename = f"{text}_{i:05d}.png"
        filepath = output_dir / filename
        image.save(filepath)
        
        metadata.append({
            "filename": filename,
            "text": text,
            "length": len(text),
            "noise": round(noise, 2),
        })
    
    # Splits si demandé
    if args.split:
        print("\n📂 Création des splits...")
        
        random.shuffle(metadata)
        n = len(metadata)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        
        splits = {
            "train": metadata[:train_end],
            "val": metadata[train_end:val_end],
            "test": metadata[val_end:],
        }
        
        for split_name, split_data in splits.items():
            split_dir = output_dir / split_name
            split_dir.mkdir(exist_ok=True)
            
            for item in split_data:
                src = output_dir / item["filename"]
                dst = split_dir / item["filename"]
                src.rename(dst)
            
            # Metadata du split
            with open(split_dir / "metadata.json", "w") as f:
                json.dump(split_data, f, indent=2)
            
            print(f"   {split_name}: {len(split_data)} images")
    
    # Sauvegarder metadata global
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump({
            "total": args.n,
            "charset": charset,
            "length": args.length,
            "dimensions": {"width": args.width, "height": args.height},
            "samples": metadata if not args.split else None,
        }, f, indent=2)
    
    print(f"\n✅ Dataset généré: {output_dir}")
    print(f"   Metadata: {metadata_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())