#!/usr/bin/env python3
"""
Benchmark Script - Benchmark des modèles en ligne de commande.

Usage:
    python scripts/benchmark.py --n-samples 50 --models trocr,crnn
    python scripts/benchmark.py --noise 0.5 --output results.json

M2 MoSEF - Université Paris 1 Panthéon-Sorbonne
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Benchmark des modèles CAPTCHA")
    parser.add_argument("--n-samples", type=int, default=20, help="Nombre d'échantillons")
    parser.add_argument("--noise", type=float, default=0.3, help="Niveau de bruit (0-1)")
    parser.add_argument("--models", type=str, default=None, help="Modèles à tester (ex: trocr,crnn)")
    parser.add_argument("--output", type=str, default=None, help="Fichier de sortie JSON")
    parser.add_argument("--verbose", action="store_true", help="Mode verbeux")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔓 CAPTCHA Factory - Benchmark")
    print("=" * 60)
    print(f"📊 Échantillons: {args.n_samples}")
    print(f"🔊 Niveau de bruit: {args.noise}")
    print()
    
    from app.services.solver_service import SolverService
    
    solver = SolverService()
    models_list = args.models.split(",") if args.models else None
    
    print(f"🤖 Modèles: {models_list or solver.available_models}")
    print()
    print("⏳ Benchmark en cours...")
    
    start_time = time.time()
    results = solver.benchmark(
        n_samples=args.n_samples,
        noise_level=args.noise,
        models=models_list,
    )
    elapsed = time.time() - start_time
    
    print(f"\n✅ Terminé en {elapsed:.1f}s\n")
    
    # Afficher les résultats
    print("=" * 60)
    print("📊 RÉSULTATS")
    print("=" * 60)
    
    for name, result in results.items():
        print(f"\n🤖 {name.upper()}")
        print(f"   Accuracy: {result.accuracy:.1f}%")
        print(f"   Corrects: {result.correct}/{result.total_samples}")
        print(f"   Temps moyen: {result.avg_time_ms:.0f}ms")
    
    # Meilleur modèle
    best = max(results.items(), key=lambda x: (x[1].accuracy, -x[1].avg_time_ms))
    print(f"\n🏆 Meilleur modèle: {best[0].upper()} ({best[1].accuracy:.1f}%)")
    
    # Sauvegarder si demandé
    if args.output:
        output_data = {
            "config": {
                "n_samples": args.n_samples,
                "noise_level": args.noise,
                "models": models_list,
            },
            "results": {name: res.to_dict() for name, res in results.items()},
            "elapsed_seconds": elapsed,
        }
        
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Résultats sauvegardés: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())