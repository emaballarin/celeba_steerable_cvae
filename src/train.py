#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple

import torch as th
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
TRAIN_BS: int = 1024
TEST_BS: int = 32
LATENT_SIZE: int = 128
CONDITION_SIZE: int = 40
EPOCHS: int = 200
BASE_LR: float = 1e-3

device = th.device("cuda" if (th.cuda.is_available() and DEVICE_AUTODETECT) else "cpu")

train_ds = CelebA(
    root="../data/",
    split="train",
    target_type="attr",
    transform=Compose([Resize(IMG_SHAPE[1:]), ToTensor()]),
    download=False,
)

test_ds = CelebA(
    root="../data/",
    split="test",
    target_type="attr",
    transform=Compose([Resize(IMG_SHAPE[1:]), ToTensor()]),
    download=False,
)

train_dl = DataLoader(
    train_ds,
    batch_size=TRAIN_BS,
    shuffle=True,
    num_workers=16,
    pin_memory=(device == th.device("cuda")),
)
test_dl = DataLoader(
    test_ds,
    batch_size=TEST_BS,
    shuffle=True,
    num_workers=4,
    pin_memory=(device == th.device("cuda")),
)

model = CelebACVAE(lat_size=LATENT_SIZE, cond_size=CONDITION_SIZE).to(device)

optimizer = RAdam(model.parameters(), lr=BASE_LR)

optimizer, scheduler = warmed_up_linneal(
    optim=optimizer,
    init_lr=BASE_LR / 1e4,
    steady_lr=BASE_LR,
    final_lr=BASE_LR / 1e5,
    warmup_epochs=5,
    steady_epochs=145,
    anneal_epochs=50,
)

beta_epochs = min(EPOCHS // 3, 75)

model.train()
for epoch in trange(EPOCHS, leave=True, desc="Epoch"):
    beta: float = (0.5 * (epoch / beta_epochs)) if epoch < beta_epochs else 0.5
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


save_model(model, "./celeba_cvae.safetensors")
