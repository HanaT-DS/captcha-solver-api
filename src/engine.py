from typing import Dict, List, Tuple, Any

from tqdm import tqdm
import torch

from src import config



def train_fn(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Entraîne le modèle sur 1 epoch et retourne la loss moyenne."""
    model.train()                        # Mode entraînement (dropout/BN actifs)
    fin_loss: float = 0.0

    tk0 = tqdm(data_loader, total=len(data_loader))  # Barre de progression

    for data in tk0:
        # Envoie chaque tensor du batch sur le bon device (cpu/cuda)
        for key, value in data.items():
            data[key] = value.to(config.DEVICE)

        optimizer.zero_grad()            # Reset des gradients

        _, loss = model(**data)          # Forward : preds + loss (le modèle calcule la loss)
        loss.backward()                  # Backprop : calcule les gradients
        optimizer.step()                 # Mise à jour des poids

        fin_loss += loss.item()          # Ajoute la loss du batch (float)

    return fin_loss / len(data_loader)   # Loss moyenne sur l'epoch


@torch.no_grad()
def eval_fn(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
) -> Tuple[List[torch.Tensor], float]:
    """Évalue le modèle et retourne (liste_preds, loss_moyenne)."""
    model.eval()                         # Mode évaluation (dropout off)
    fin_loss: float = 0.0
    fin_preds: List[torch.Tensor] = []

    tk0 = tqdm(data_loader, total=len(data_loader))

    for data in tk0:
        # Envoie batch sur le bon device
        for key, value in data.items():
            data[key] = value.to(config.DEVICE)

        batch_preds, loss = model(**data)  # Forward seulement (pas de backprop)
        fin_loss += loss.item()
        fin_preds.append(batch_preds)      # Garde les prédictions brutes

    return fin_preds, fin_loss / len(data_loader)
