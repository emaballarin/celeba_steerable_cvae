#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple

import torch as th
import wandb
from losses import beta_reco_bce_splitout
from losses import var_of_lap
from optim import RAdam
from safetensors.torch import save_model
from schedulers import warmed_up_linneal
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm
from tqdm.auto import trange

from models import CelebACVAE

DEVICE_AUTODETECT: bool = True
IMG_SHAPE: Tuple[int, int, int] = (3, 64, 64)
TRAIN_BS: int = 256
LATENT_SIZE: int = 128
CONDITION_SIZE: int = 14
EPOCHS: int = 200
BASE_LR: float = 1e-3
BETA_LAG: int = 5
BETA_EPOCHS: int = max(0, EPOCHS // 10 - BETA_LAG)
BETA_MAX: float = 6.0  # Target: RL/KL ~ 10
VOL_SCALE: float = 5000000 / TRAIN_BS  # Target: RL/SCVOL ~ 20

device = th.device("cuda" if (th.cuda.is_available() and DEVICE_AUTODETECT) else "cpu")


dl_opt_kwargs = (
    {
        "pin_memory": (device == th.device("cuda")),
        "pin_memory_device": (
            "cuda" if (device == th.device("cuda")) else ""  # NOSONAR
        ),
    }
    if TRAIN_BS > 1024
    else {}
)

train_ds = CelebA(
    root="../data/",
    split="train",
    target_type="attr",
    transform=Compose([Resize(IMG_SHAPE[1:]), ToTensor()]),
    download=False,
)

train_dl = DataLoader(
    train_ds,
    batch_size=TRAIN_BS,
    shuffle=True,
    num_workers=16,
    persistent_workers=True,
    **dl_opt_kwargs,
)

model = CelebACVAE(lat_size=LATENT_SIZE, cond_size=CONDITION_SIZE).to(device)

optimizer = RAdam(model.parameters(), lr=BASE_LR, betas=(0.9, 0.985))  # Dieleman, 2023

optimizer, scheduler = warmed_up_linneal(
    optim=optimizer,
    init_lr=BASE_LR * 1e-4,  # Empirical
    steady_lr=BASE_LR,  # Standard practice
    final_lr=BASE_LR * 1e-3,  # For ~long annealing
    warmup_epochs=4,  # Minimum possible
    steady_epochs=2 * (EPOCHS - 4) // 3,
    anneal_epochs=(EPOCHS - 4) // 3,  # Quench a little
)

loss: th.Tensor = th.tensor(0.0, device=device)
reco: th.Tensor = th.tensor(0.0, device=device)
kldiv: th.Tensor = th.tensor(0.0, device=device)
scvol: th.Tensor = th.tensor(0.0, device=device)

wandb.init(
    project="celeba_sweeping_cvae",
    config={
        "version": "v11",
    },
)

model.train()
for epoch in trange(EPOCHS, leave=True, desc="Epoch"):
    if epoch < BETA_LAG:
        beta: float = 0.0
    else:
        if BETA_EPOCHS > 0:
            beta: float = (
                ((epoch - BETA_LAG) / BETA_EPOCHS)
                if (epoch - BETA_LAG) < BETA_EPOCHS
                else 1.0
            ) * BETA_MAX
        else:
            beta = BETA_MAX

    for i, (images, attr) in tqdm(
        enumerate(train_dl), total=len(train_dl), leave=False, desc="Batch"
    ):
        images: th.Tensor = images.to(device)
        attr: th.Tensor = th.index_select(
            attr, 1, th.tensor([0, 4, 5, 8, 9, 10, 15, 18, 20, 24, 25, 28, 31, 39])
        )
        attr: th.Tensor = attr.to(device)
        optimizer.zero_grad()
        reconstructed_images, mean, log_var = model(images, attr)
        loss, reco, kldiv = beta_reco_bce_splitout(
            reconstructed_images, images, mean, log_var, beta
        )
        scvol = var_of_lap(reconstructed_images).sum() * VOL_SCALE
        loss = loss + scvol
        loss.backward()
        optimizer.step()
    scheduler.step()
    wandb.log(
        {
            "lossT": loss.item(),
            "lossR": reco.item(),
            "lossK": kldiv.item(),
            "lossV": scvol.item(),
            "beta": beta,
            "lr": scheduler.get_last_lr()[0],
        },
        step=epoch,
    )
wandb.finish()


save_model(model, "./celeba_cvae_v11.safetensors")
