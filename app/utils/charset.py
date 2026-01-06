"""
Charset Utilities - Gestion des jeux de caractères.

Fonctions pour:
- Validation de charset
- Vérification de compatibilité
- Normalisation de texte
- Calcul de métriques (CER, Accuracy)

M2 MoSEF - Université Paris 1 Panthéon-Sorbonne
"""

import re
from typing import Set, List, Optional, Tuple
from difflib import SequenceMatcher


# =============================================================================
# CHARSETS PRÉDÉFINIS
# =============================================================================

CHARSET_CRNN = "23456789bcdefgmnpwxy"  # 19 caractères (dataset original)
CHARSET_DIGITS = "0123456789"  # 10 chiffres
CHARSET_LOWERCASE = "abcdefghijklmnopqrstuvwxyz"  # 26 lettres minuscules
CHARSET_UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 26 lettres majuscules
CHARSET_ALPHANUMERIC = CHARSET_DIGITS + CHARSET_LOWERCASE + CHARSET_UPPERCASE  # 62 caractères

# Caractères confusants (visuellement similaires)
CONFUSING_CHARS = {
    '0': 'O', 'O': '0',
    '1': 'l', 'l': '1',
    'I': 'l', 'l': 'I',
    '5': 'S', 'S': '5',
    '8': 'B', 'B': '8',
}


# =============================================================================
# VALIDATION
# =============================================================================

def validate_charset(charset: str) -> Tuple[bool, Optional[str]]:
    """
    Valide un charset.
    
    Args:
        charset: Chaîne de caractères à valider.
        
    Returns:
        Tuple (is_valid, error_message).
    """
    if not charset:
        return False, "Charset vide"
    
    if len(charset) < 2:
        return False, "Charset trop court (minimum 2 caractères)"
    
    # Vérifier les doublons
    if len(set(charset)) != len(charset):
        duplicates = [c for c in charset if charset.count(c) > 1]
        return False, f"Caractères en double: {set(duplicates)}"
    
    # Vérifier que ce sont des caractères imprimables
    for c in charset:
        if not c.isprintable():
            return False, f"Caractère non imprimable: {repr(c)}"
    
    return True, None


def is_charset_compatible(
    text: str,
    charset: str,
    case_sensitive: bool = False,
) -> bool:
    """
    Vérifie si un texte est compatible avec un charset.
    
    Args:
        text: Texte à vérifier.
        charset: Charset cible.
        case_sensitive: Si False, ignore la casse.
        
    Returns:
        True si tous les caractères sont dans le charset.
    """
    if not case_sensitive:
        text = text.lower()
        charset = charset.lower()
    
    charset_set = set(charset)
    return all(c in charset_set for c in text)


def get_incompatible_chars(
    text: str,
    charset: str,
    case_sensitive: bool = False,
) -> Set[str]:
    """
    Retourne les caractères incompatibles.
    
    Args:
        text: Texte à vérifier.
        charset: Charset cible.
        case_sensitive: Si False, ignore la casse.
        
    Returns:
        Ensemble des caractères non supportés.
    """
    if not case_sensitive:
        text = text.lower()
        charset = charset.lower()
    
    charset_set = set(charset)
    return {c for c in text if c not in charset_set}


# =============================================================================
# NORMALISATION
# =============================================================================

def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_spaces: bool = True,
    alphanumeric_only: bool = True,
) -> str:
    """
    Normalise un texte pour la comparaison.
    
    Args:
        text: Texte à normaliser.
        lowercase: Convertir en minuscules.
        remove_spaces: Supprimer les espaces.
        alphanumeric_only: Garder seulement alphanumériques.
        
    Returns:
        Texte normalisé.
    """
    if lowercase:
        text = text.lower()
    
    if remove_spaces:
        text = text.replace(" ", "")
    
    if alphanumeric_only:
        text = re.sub(r'[^a-zA-Z0-9]', '', text)
    
    return text


def filter_to_charset(
    text: str,
    charset: str,
) -> str:
    """
    Filtre un texte pour ne garder que les caractères du charset.
    
    Args:
        text: Texte à filtrer.
        charset: Charset autorisé.
        
    Returns:
        Texte filtré.
    """
    charset_set = set(charset)
    return "".join(c for c in text if c in charset_set)


def replace_confusing_chars(
    text: str,
    direction: str = "digit_to_letter",
) -> str:
    """
    Remplace les caractères confusants.
    
    Args:
        text: Texte à traiter.
        direction: 'digit_to_letter' ou 'letter_to_digit'.
        
    Returns:
        Texte avec remplacements.
    """
    if direction == "digit_to_letter":
        replacements = {'0': 'O', '1': 'l', '5': 'S', '8': 'B'}
    else:
        replacements = {'O': '0', 'l': '1', 'S': '5', 'B': '8', 'I': '1'}
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


# =============================================================================
# MÉTRIQUES
# =============================================================================

def compute_cer(
    predicted: str,
    ground_truth: str,
    normalize: bool = True,
) -> float:
    """
    Calcule le Character Error Rate (CER).
    
    CER = (Substitutions + Insertions + Deletions) / Longueur référence
    
    Args:
        predicted: Texte prédit.
        ground_truth: Texte de référence.
        normalize: Si True, normalise les textes avant comparaison.
        
    Returns:
        CER entre 0 et 1 (ou plus si predicted beaucoup plus long).
    """
    if normalize:
        predicted = normalize_text(predicted)
        ground_truth = normalize_text(ground_truth)
    
    if not ground_truth:
        return 0.0 if not predicted else 1.0
    
    # Distance de Levenshtein
    distance = levenshtein_distance(predicted, ground_truth)
    
    return distance / len(ground_truth)


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calcule la distance de Levenshtein entre deux chaînes.
    
    Args:
        s1: Première chaîne.
        s2: Deuxième chaîne.
        
    Returns:
        Nombre minimum d'opérations.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def compute_accuracy(
    predicted: str,
    ground_truth: str,
    normalize: bool = True,
    case_sensitive: bool = False,
) -> float:
    """
    Calcule l'accuracy (1 si exact match, 0 sinon).
    
    Args:
        predicted: Texte prédit.
        ground_truth: Texte de référence.
        normalize: Si True, normalise les textes.
        case_sensitive: Si True, compare avec casse.
        
    Returns:
        1.0 si match exact, 0.0 sinon.
    """
    if normalize:
        predicted = normalize_text(predicted)
        ground_truth = normalize_text(ground_truth)
    
    if not case_sensitive:
        predicted = predicted.lower()
        ground_truth = ground_truth.lower()
    
    return 1.0 if predicted == ground_truth else 0.0


def compute_char_accuracy(
    predicted: str,
    ground_truth: str,
    normalize: bool = True,
) -> float:
    """
    Calcule l'accuracy au niveau caractère.
    
    Args:
        predicted: Texte prédit.
        ground_truth: Texte de référence.
        normalize: Si True, normalise les textes.
        
    Returns:
        Ratio de caractères corrects.
    """
    if normalize:
        predicted = normalize_text(predicted)
        ground_truth = normalize_text(ground_truth)
    
    if not ground_truth:
        return 1.0 if not predicted else 0.0
    
    # Utiliser SequenceMatcher pour l'alignement
    matcher = SequenceMatcher(None, predicted, ground_truth)
    matches = sum(block.size for block in matcher.get_matching_blocks())
    
    return matches / max(len(predicted), len(ground_truth))


def compute_word_accuracy(
    predictions: List[str],
    ground_truths: List[str],
    normalize: bool = True,
) -> float:
    """
    Calcule l'accuracy sur un batch.
    
    Args:
        predictions: Liste des prédictions.
        ground_truths: Liste des références.
        normalize: Si True, normalise les textes.
        
    Returns:
        Ratio de prédictions exactes.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Les listes doivent avoir la même longueur")
    
    if not predictions:
        return 0.0
    
    correct = sum(
        compute_accuracy(p, g, normalize)
        for p, g in zip(predictions, ground_truths)
    )
    
    return correct / len(predictions)


# =============================================================================
# ANALYSE
# =============================================================================

def analyze_errors(
    predicted: str,
    ground_truth: str,
) -> dict:
    """
    Analyse les erreurs entre prédiction et référence.
    
    Args:
        predicted: Texte prédit.
        ground_truth: Texte de référence.
        
    Returns:
        Dictionnaire avec l'analyse détaillée.
    """
    predicted = normalize_text(predicted)
    ground_truth = normalize_text(ground_truth)
    
    matcher = SequenceMatcher(None, predicted, ground_truth)
    opcodes = matcher.get_opcodes()
    
    errors = {
        "substitutions": [],
        "insertions": [],
        "deletions": [],
    }
    
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "replace":
            errors["substitutions"].append({
                "predicted": predicted[i1:i2],
                "expected": ground_truth[j1:j2],
                "position": j1,
            })
        elif tag == "insert":
            errors["insertions"].append({
                "chars": predicted[i1:i2],
                "position": j1,
            })
        elif tag == "delete":
            errors["deletions"].append({
                "chars": ground_truth[j1:j2],
                "position": j1,
            })
    
    return {
        "predicted": predicted,
        "ground_truth": ground_truth,
        "exact_match": predicted == ground_truth,
        "cer": compute_cer(predicted, ground_truth, normalize=False),
        "errors": errors,
        "total_errors": sum(len(v) for v in errors.values()),
    }