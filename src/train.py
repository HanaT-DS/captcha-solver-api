import os
import glob
from typing import List

import torch
import numpy as np

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

from src import config
from src import dataset
from src import engine
from src.model import CaptchaModel



def remove_duplicates(x: str) -> str:
    """Supprime les caractères consécutifs identiques (CTC)"""
    if len(x) < 2:
        return x

    fin: str = ""
    for j in x:
        if fin == "":
            fin = j
        else:
            if j == fin[-1]:
                continue
            else:
                fin = fin + j
    return fin


def decode_predictions(
    preds: torch.Tensor,
    encoder: preprocessing.LabelEncoder
) -> List[str]:
    """Convertit les sorties du modèle en texte lisible"""
    preds = preds.permute(1, 0, 2)        # (T, B, C) -> (B, T, C)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()

    cap_preds: List[str] = []

    for j in range(preds.shape[0]):
        temp: List[str] = []
        for k in preds[j, :]:
            k = k - 1                     # Décalage (0 = blank)
            if k == -1:
                temp.append("§")
            else:
                p: str = encoder.inverse_transform([k])[0]
                temp.append(p)

        tp: str = "".join(temp).replace("§", "")
        cap_preds.append(remove_duplicates(tp))

    return cap_preds


def run_training() -> None:
    """Lance l'entraînement du modèle CAPTCHA"""

    # Chargement des images
    image_files: List[str] = glob.glob(
        os.path.join(config.DATA_DIR, "*.png")
    )

    # Labels = noms des fichiers sans extension
    targets_orig: List[str] = [
        os.path.splitext(os.path.basename(x))[0] for x in image_files
    ]

    # Chaque captcha devient une liste de caractères
    targets: List[List[str]] = [[c for c in x] for x in targets_orig]

    # Aplatit tous les caractères
    targets_flat: List[str] = [
        c for clist in targets for c in clist
    ]

    # Encodeur de caractères
    lbl_enc: preprocessing.LabelEncoder = preprocessing.LabelEncoder()
    lbl_enc.fit(targets_flat)

    # Encodage des labels
    targets_enc = [lbl_enc.transform(x) for x in targets]
    targets_enc = np.array(targets_enc)
    targets_enc = targets_enc + 1          # 0 réservé au blank

    # Split train / test
    (
        train_imgs,
        test_imgs,
        train_targets,
        test_targets,
        _,
        test_targets_orig,
    ) = model_selection.train_test_split(
        image_files,
        targets_enc,
        targets_orig,
        test_size=0.1,
        random_state=42,
    )

    # Dataset train
    train_dataset = dataset.ClassificationDataset(
        image_paths=train_imgs,
        targets=train_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
    )

    # Dataset test
    test_dataset = dataset.ClassificationDataset(
        image_paths=test_imgs,
        targets=test_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
    )

    # Modèle
    model: CaptchaModel = CaptchaModel(
        num_chars=len(lbl_enc.classes_)
    )
    model.to(config.DEVICE)

    # Optimiseur
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )

    # Entraînement
    for epoch in range(config.EPOCHS):
        train_loss: float = engine.train_fn(
            model, train_loader, optimizer
        )
        valid_preds, test_loss = engine.eval_fn(
            model, test_loader
        )

        # Décodage des prédictions
        valid_captcha_preds: List[str] = []
        for vp in valid_preds:
            valid_captcha_preds.extend(
                decode_predictions(vp, lbl_enc)
            )

        # Accuracy
        test_dup_rem: List[str] = [
            remove_duplicates(c) for c in test_targets_orig
        ]
        accuracy: float = metrics.accuracy_score(
            test_dup_rem, valid_captcha_preds
        )

        print(
            f"Epoch={epoch} | "
            f"Train Loss={train_loss:.4f} | "
            f"Test Loss={test_loss:.4f} | "
            f"Accuracy={accuracy:.4f}"
        )

        scheduler.step(test_loss)


if __name__ == "__main__":
    run_training()





