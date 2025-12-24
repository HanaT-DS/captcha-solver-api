from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class CaptchaModel(nn.Module):
    def __init__(self, num_chars: int) -> None:
        super().__init__()

        # 2 blocs convolution + pooling (extraction de features)
        self.conv_1: nn.Conv2d = nn.Conv2d(3, 128, kernel_size=(3, 6), padding=(1, 1))
        self.pool_1: nn.MaxPool2d = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv_2: nn.Conv2d = nn.Conv2d(128, 64, kernel_size=(3, 6), padding=(1, 1))
        self.pool_2: nn.MaxPool2d = nn.MaxPool2d(kernel_size=(2, 2))

        # Projection linéaire (features -> 64)
        self.linear_1: nn.Linear = nn.Linear(1152, 64)
        self.drop_1: nn.Dropout = nn.Dropout(0.2)

        # GRU bidirectionnel (séquence)
        self.lstm: nn.GRU = nn.GRU(
            input_size=64,
            hidden_size=32,
            bidirectional=True,
            num_layers=2,
            dropout=0.25,
            batch_first=True,
        )

        # Couche de sortie : num_chars + 1 (le +1 = blank CTC)
        self.output: nn.Linear = nn.Linear(64, num_chars + 1)

        # Loss CTC (créée une seule fois)
        self.ctc_loss: nn.CTCLoss = nn.CTCLoss(blank=0)

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        images  : Tensor (B, 3, H, W)
        targets : Tensor (B, seq_len) contenant les labels encodés (optionnel)
        return  : (logits, loss) -> loss est None si targets=None
        """

        bs: int = images.size(0)  # batch size

        # Conv block 1
        x: torch.Tensor = F.relu(self.conv_1(images))
        x = self.pool_1(x)

        # Conv block 2
        x = F.relu(self.conv_2(x))
        x = self.pool_2(x)

        # Préparer pour RNN : on met la largeur W comme dimension "time"
        x = x.permute(0, 3, 1, 2)         # (B, C, H, W) -> (B, W, C, H)
        x = x.view(bs, x.size(1), -1)     # (B, W, C*H)

        # Linear + dropout
        x = F.relu(self.linear_1(x))      # (B, W, 64)
        x = self.drop_1(x)

        # GRU
        x, _ = self.lstm(x)               # (B, W, 64) car biGRU (32*2)

        # Sortie logits
        x = self.output(x)                # (B, W, num_chars+1)
        x = x.permute(1, 0, 2)            # (W, B, C) format attendu par CTC

        # Si on a targets -> calcul de la loss
        if targets is not None:
            log_probs: torch.Tensor = F.log_softmax(x, dim=2)

            # Longueur des séquences en entrée (toutes égales à W ici)
            input_lengths: torch.Tensor = torch.full(
                size=(bs,),
                fill_value=log_probs.size(0),
                dtype=torch.int32,
                device=log_probs.device,
            )

            # Longueur des targets (toutes égales à seq_len)
            target_lengths: torch.Tensor = torch.full(
                size=(bs,),
                fill_value=targets.size(1),
                dtype=torch.int32,
                device=targets.device,
            )

            loss: torch.Tensor = self.ctc_loss(
                log_probs, targets, input_lengths, target_lengths
            )
            return x, loss

        return x, None


if __name__ == "__main__":
    # Petit test rapide
    cm = CaptchaModel(num_chars=19)

    # Exemple : image (B=1, 3, H=75, W=300) pour matcher ton config
    img = torch.rand((1, 3, 75, 300))

    # Targets doivent être des entiers (dtype long), pas des floats
    targets = torch.randint(low=1, high=20, size=(1, 5), dtype=torch.long)

    logits, loss = cm(img, targets)
    print("logits shape:", logits.shape)
    print("loss:", loss.item() if loss is not None else None)
