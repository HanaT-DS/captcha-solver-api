"""
Confidence Utilities - Calcul et calibration de la confiance.

Fonctions pour:
- Calculer la confiance des prédictions CTC
- Calculer la confiance des séquences autoregressives
- Calibrer les scores de confiance
- Vérifier les seuils de confiance

M2 MoSEF - Université Paris 1 Panthéon-Sorbonne
"""

import numpy as np
from typing import List, Optional, Tuple, Union


# =============================================================================
# CONFIANCE CTC (CRNN)
# =============================================================================

def compute_ctc_confidence(
    log_probs: np.ndarray,
    predicted_indices: np.ndarray,
    blank_idx: int = 0,
) -> float:
    """
    Calcule la confiance pour une prédiction CTC.
    
    La confiance est calculée comme la moyenne géométrique des
    probabilités des tokens non-blank prédits.
    
    Args:
        log_probs: Log-probabilités de sortie (T, C).
        predicted_indices: Indices prédits par greedy decoding.
        blank_idx: Index du token blank.
        
    Returns:
        Score de confiance entre 0 et 1.
    """
    # Convertir en probabilités
    probs = np.exp(log_probs)
    
    # Collecter les probabilités des tokens non-blank
    token_probs = []
    prev_idx = blank_idx
    
    for t, idx in enumerate(predicted_indices):
        if idx != blank_idx and idx != prev_idx:
            token_probs.append(probs[t, idx])
        prev_idx = idx
    
    if not token_probs:
        return 0.0
    
    # Moyenne géométrique
    confidence = np.exp(np.mean(np.log(token_probs)))
    
    return float(np.clip(confidence, 0, 1))


def compute_ctc_confidence_with_beam(
    log_probs: np.ndarray,
    beam_width: int = 5,
) -> Tuple[str, float]:
    """
    Calcule la confiance avec beam search.
    
    Plus robuste que le greedy decoding mais plus lent.
    
    Args:
        log_probs: Log-probabilités (T, C).
        beam_width: Largeur du beam.
        
    Returns:
        Tuple (séquence décodée, confiance).
    """
    # Implémentation simplifiée du beam search
    T, C = log_probs.shape
    
    # Initialisation
    beams = [([], 0.0)]  # (séquence, score)
    
    for t in range(T):
        all_candidates = []
        
        for seq, score in beams:
            # Étendre avec chaque token
            for c in range(C):
                new_score = score + log_probs[t, c]
                new_seq = seq + [c]
                all_candidates.append((new_seq, new_score))
        
        # Garder les meilleurs
        beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    
    # Meilleur résultat
    best_seq, best_score = beams[0]
    
    # Confidence normalisée par la longueur
    confidence = np.exp(best_score / T)
    
    return best_seq, float(confidence)


# =============================================================================
# CONFIANCE SÉQUENTIELLE (TrOCR, Florence-2)
# =============================================================================

def compute_sequence_confidence(
    token_probs: List[float],
    method: str = "geometric",
) -> float:
    """
    Calcule la confiance d'une séquence autoregressif.
    
    Args:
        token_probs: Liste des probabilités par token.
        method: Méthode de calcul:
            - 'geometric': Moyenne géométrique (défaut)
            - 'arithmetic': Moyenne arithmétique
            - 'minimum': Probabilité minimale
            - 'product': Produit (très sensible)
            
    Returns:
        Score de confiance entre 0 et 1.
    """
    if not token_probs:
        return 0.0
    
    probs = np.array(token_probs)
    probs = np.clip(probs, 1e-10, 1.0)  # Éviter log(0)
    
    if method == "geometric":
        # Moyenne géométrique = exp(mean(log(probs)))
        confidence = np.exp(np.mean(np.log(probs)))
    
    elif method == "arithmetic":
        confidence = np.mean(probs)
    
    elif method == "minimum":
        confidence = np.min(probs)
    
    elif method == "product":
        confidence = np.prod(probs)
    
    else:
        raise ValueError(f"Méthode inconnue: {method}")
    
    return float(np.clip(confidence, 0, 1))


def extract_token_probabilities(
    scores: List[np.ndarray],
    generated_ids: np.ndarray,
    eos_token_id: int,
) -> List[float]:
    """
    Extrait les probabilités des tokens générés.
    
    Args:
        scores: Liste des logits par step (de model.generate()).
        generated_ids: IDs des tokens générés.
        eos_token_id: ID du token de fin.
        
    Returns:
        Liste des probabilités.
    """
    probabilities = []
    
    for step_idx, (score, token_id) in enumerate(zip(scores, generated_ids[1:])):
        if token_id == eos_token_id:
            break
        
        # Softmax
        probs = np.exp(score) / np.sum(np.exp(score))
        token_prob = probs[token_id]
        probabilities.append(float(token_prob))
    
    return probabilities


# =============================================================================
# CALIBRATION
# =============================================================================

def calibrate_confidence(
    raw_confidence: float,
    model_type: str = "crnn",
    temperature: float = 1.0,
) -> float:
    """
    Calibre la confiance brute pour être plus fiable.
    
    Les modèles ont tendance à être sur-confiants. Cette fonction
    applique une calibration pour améliorer la fiabilité.
    
    Args:
        raw_confidence: Confiance brute du modèle.
        model_type: Type de modèle ('crnn', 'trocr', 'florence').
        temperature: Température de calibration (>1 réduit la confiance).
        
    Returns:
        Confiance calibrée.
    """
    # Températures par défaut par modèle
    default_temps = {
        "crnn": 1.2,      # CRNN légèrement sur-confiant
        "trocr": 1.1,     # TrOCR bien calibré
        "florence": 1.5,  # Florence-2 souvent sur-confiant en OCR
        "easyocr": 1.3,   # EasyOCR variable
    }
    
    if temperature == 1.0 and model_type in default_temps:
        temperature = default_temps[model_type]
    
    # Appliquer la température via logit scaling
    if raw_confidence <= 0 or raw_confidence >= 1:
        return raw_confidence
    
    # Convertir en logit
    logit = np.log(raw_confidence / (1 - raw_confidence))
    
    # Appliquer la température
    scaled_logit = logit / temperature
    
    # Reconvertir en probabilité
    calibrated = 1 / (1 + np.exp(-scaled_logit))
    
    return float(np.clip(calibrated, 0, 1))


def isotonic_calibration(
    confidences: List[float],
    accuracies: List[bool],
) -> callable:
    """
    Entraîne un calibrateur isotonique.
    
    Utile pour calibrer sur un dataset de validation.
    
    Args:
        confidences: Confiances brutes.
        accuracies: Indicateurs de correction (True/False).
        
    Returns:
        Fonction de calibration.
    """
    try:
        from sklearn.isotonic import IsotonicRegression
        
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(confidences, accuracies)
        
        return lambda x: ir.predict([x])[0]
    
    except ImportError:
        # Fallback: pas de calibration
        return lambda x: x


# =============================================================================
# SEUILS
# =============================================================================

def confidence_threshold_check(
    confidence: Optional[float],
    threshold: float = 0.85,
    strict: bool = False,
) -> bool:
    """
    Vérifie si la confiance dépasse le seuil.
    
    Args:
        confidence: Score de confiance (peut être None).
        threshold: Seuil minimum.
        strict: Si True, retourne False pour None.
        
    Returns:
        True si confiance >= seuil.
    """
    if confidence is None:
        return not strict
    
    return confidence >= threshold


def get_confidence_level(
    confidence: Optional[float],
) -> str:
    """
    Retourne un niveau de confiance textuel.
    
    Args:
        confidence: Score de confiance.
        
    Returns:
        Niveau: 'high', 'medium', 'low', 'unknown'.
    """
    if confidence is None:
        return "unknown"
    
    if confidence >= 0.9:
        return "high"
    elif confidence >= 0.7:
        return "medium"
    else:
        return "low"


def recommend_model_by_confidence(
    confidences: dict,
    threshold: float = 0.85,
) -> Optional[str]:
    """
    Recommande un modèle basé sur les confiances.
    
    Args:
        confidences: Dict {model_name: confidence}.
        threshold: Seuil minimum acceptable.
        
    Returns:
        Nom du modèle recommandé ou None.
    """
    # Filtrer les modèles au-dessus du seuil
    valid = {k: v for k, v in confidences.items() if v and v >= threshold}
    
    if not valid:
        return None
    
    # Retourner celui avec la plus haute confiance
    return max(valid, key=valid.get)