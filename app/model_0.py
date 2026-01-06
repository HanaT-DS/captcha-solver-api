"""
Définition de l'architecture CRNN pour la résolution de CAPTCHAs.
"""
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Optional


class CaptchaModel(nn.Module):
    """
    Modèle hybride CRNN (CNN + RNN) pour la reconnaissance de caractères.
    """
    def __init__(self, num_chars: int) -> None:
        """
        Initialise les couches du modèle.
        
        Args:
            num_chars: Nombre de caractères uniques dans le dictionnaire.
        """
        super().__init__()

        # 1. Extraction de caractéristiques (CNN)
        # Bloc 1 : Détecte les formes simples
        self.conv_1: nn.Conv2d = nn.Conv2d(3, 128, kernel_size=(3, 6), padding=(1, 1))
        self.pool_1: nn.MaxPool2d = nn.MaxPool2d(kernel_size=(2, 2))

        # Bloc 2 : Détecte les formes complexes
        self.conv_2: nn.Conv2d = nn.Conv2d(128, 64, kernel_size=(3, 6), padding=(1, 1))
        self.pool_2: nn.MaxPool2d = nn.MaxPool2d(kernel_size=(2, 2))

        # 2. Transition vers le RNN
        # Réduit la dimension avant d'entrer dans la séquence
        self.linear_1: nn.Linear = nn.Linear(1152, 64)
        self.drop_1: nn.Dropout = nn.Dropout(0.2)

        # 3. Analyse de la séquence (RNN)
        # Utilise un GRU bidirectionnel pour lire de gauche à droite ET de droite à gauche
        self.gru: nn.GRU = nn.GRU(  
            input_size=64,
            hidden_size=32,
            bidirectional=True,
            num_layers=2,
            dropout=0.25,
            batch_first=True,
        )

        # 4. Couche de sortie
        # num_chars + 1 pour inclure le caractère spécial "blank"
        self.output: nn.Linear = nn.Linear(64, num_chars + 1)

    def forward(
        self, 
        images: torch.Tensor, 
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Définit le passage des données dans le réseau.
        
        Args:
            images: Batch d'images de taille (B, 3, H, W).
            targets: Labels encodés (optionnel).
            
        Returns:
            Logits (prédictions brutes) et la perte (loss) si targets est fourni.
        """
        bs: int = images.size(0)
        device: torch.device = images.device

        # Passage dans les convolutions
        x: torch.Tensor = F.relu(self.conv_1(images))
        x = self.pool_1(x)
        x = F.relu(self.conv_2(x))
        x = self.pool_2(x)

        # Préparation pour le RNN : on transforme l'image en séquence
        # On veut que la largeur (W) devienne notre dimension temporelle
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W) -> (B, W, C, H)
        x = x.view(bs, x.size(1), -1) 

        x = F.relu(self.linear_1(x))
        x = self.drop_1(x)

        # Passage dans le GRU
        x, _ = self.gru(x)  

        # Calcul des prédictions finales
        x = self.output(x)
        
        # Formatage pour la perte CTC (Largeur, Batch, Classes)
        x = x.permute(1, 0, 2)

        if targets is not None:
            # On utilise ici la CTC Loss interne
            log_probs = F.log_softmax(x, dim=2)
            input_lengths = torch.full(size=(bs,), fill_value=log_probs.size(0), dtype=torch.int32).to(device)
            target_lengths = torch.full(size=(bs,), fill_value=targets.size(1), dtype=torch.int32).to(device)
            loss = nn.CTCLoss(blank=0)(log_probs, targets, input_lengths, target_lengths)
            return x, loss

        return x, None