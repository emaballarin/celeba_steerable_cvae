#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple

import torch as th
import wandb
from losses import beta_reco_bce
from models import CelebACVAE
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

DEVICE_AUTODETECT: bool = True
IMG_SHAPE: Tuple[int, int, int] = (3, 64, 64)
TRAIN_BS: int = 256
LATENT_SIZE: int = 128
CONDITION_SIZE: int = 40
EPOCHS: int = 60
BASE_LR: float = 1e-3
BETA_EPOCHS: int = EPOCHS // 5
BETA_MAX: float = 1.0

device = th.device("cuda" if (th.cuda.is_available() and DEVICE_AUTODETECT) else "cpu")


dl_opt_kwargs = (
    {
        "pin_memory": (device == th.device("cuda")),
        "pin_memory_device": (
            "cuda" if (device == th.device("cuda")) else ""  # NOSONAR
        ),
    }
    if TRAIN_BS >= 1024
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

model = CelebACVAE(lat_size=LATENT_SIZE, cond_size=CONDITION_SIZE, shared_neck=True).to(
    device
)

optimizer = RAdam(model.parameters(), lr=BASE_LR)

optimizer, scheduler = warmed_up_linneal(
    optim=optimizer,
    init_lr=BASE_LR * 1e-4,
    steady_lr=BASE_LR,
    final_lr=BASE_LR * 1e-6,
    warmup_epochs=4,
    steady_epochs=4 * EPOCHS // 5 - 4,
    anneal_epochs=EPOCHS // 5,
)

loss: th.Tensor = th.tensor(0.0, device=device)

wandb.init(
    project="celeba_sweeping_cvae",
    config={
        "version": "v5",
    },
)

model.train()
for epoch in trange(EPOCHS, leave=True, desc="Epoch"):
    beta: float = ((epoch / BETA_EPOCHS) if epoch < BETA_EPOCHS else 1.0) * BETA_MAX
    for i, (images, attr) in tqdm(
        enumerate(train_dl), total=len(train_dl), leave=False, desc="Batch"
    ):
        images: th.Tensor = images.to(device)
        attr: th.Tensor = attr.to(device)
        optimizer.zero_grad()
        reconstructed_images, mean, log_var = model(images, attr)
        loss = beta_reco_bce(reconstructed_images, images, mean, log_var, beta)
        loss.backward()
        optimizer.step()
    scheduler.step()
    wandb.log(
        {"loss": loss.item(), "beta": beta, "lr": optimizer.param_groups[0]["lr"]},
        step=epoch,
    )
wandb.finish()


save_model(model, "./celeba_cvae_v6.safetensors")
