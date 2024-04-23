#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch as th
from torch import nn

# ──────────────────────────────────────────────────────────────────────────────


class Concatenate(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim: int = dim

    def forward(self, tensors: Union[Tuple[th.Tensor, ...], List[th.Tensor]]):
        return th.cat(tensors, dim=self.dim)


class DuplexLinearNeck(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.x_to_mu: nn.Linear = nn.Linear(in_dim, latent_dim)
        self.x_to_log_var: nn.Linear = nn.Linear(in_dim, latent_dim)

    def forward(
        self, xc: Union[Tuple[th.Tensor, ...], List[th.Tensor]]
    ) -> Tuple[th.Tensor, th.Tensor]:
        cxc: th.Tensor = th.cat(xc, dim=1)
        return self.x_to_mu(cxc), self.x_to_log_var(cxc)


class SharedDuplexLinearNeck(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.shared_layer: nn.Linear = nn.Linear(in_dim, 2 * latent_dim)

    def forward(
        self, xc: Union[Tuple[th.Tensor, ...], List[th.Tensor]]
    ) -> Tuple[th.Tensor, th.Tensor]:
        cxc: th.Tensor = th.cat(xc, dim=1)
        # noinspection PyTypeChecker
        return th.chunk(self.shared_layer(cxc), 2, dim=1)


class GaussianReparameterizerSampler(nn.Module):
    def __init__(self):
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, z_mu: th.Tensor, z_log_var: th.Tensor) -> th.Tensor:
        return z_mu + th.randn_like(z_mu, device=z_mu.device) * th.exp(z_log_var * 0.5)


def make_encoder_base() -> nn.Module:
    return nn.Sequential(
        # 3x64x64
        nn.Conv2d(3, 32, 3, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 256, 3, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(256),
        # 256x4x4
    )


def make_decoder_base(lat_size: int = 128, cond_size: int = 40) -> nn.Module:
    return nn.Sequential(
        nn.Linear(lat_size + cond_size, 256 * 4 * 4),
        nn.LeakyReLU(0.2),
        nn.Unflatten(1, (256, 4, 4)),
        nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(256),
        nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(128),
        nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(32),
        nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1),
        nn.Sigmoid(),
    )


# ──────────────────────────────────────────────────────────────────────────────


class CelebACVAE(nn.Module):
    def __init__(
        self, lat_size: int = 128, cond_size: int = 14, shared_neck: bool = False
    ):
        super().__init__()
        self.lat_size: int = lat_size
        self.cond_size: int = cond_size
        self.encoder: nn.Module = make_encoder_base()
        if shared_neck:
            self.neck: nn.Module = SharedDuplexLinearNeck(
                4096 + self.cond_size, self.lat_size
            )
        else:
            self.neck: nn.Module = DuplexLinearNeck(
                4096 + self.cond_size, self.lat_size
            )
        self.sampler: nn.Module = GaussianReparameterizerSampler()
        self.decoder: nn.Module = nn.Sequential(
            Concatenate(), make_decoder_base(self.lat_size, self.cond_size)
        )

    def forward(
        self, x: th.Tensor, c: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        compressed = self.encoder(x)
        latent_mu, latent_logvar = self.neck((th.flatten(compressed, start_dim=1), c))
        sampled = self.sampler(latent_mu, latent_logvar)
        reconstructed = self.decoder((sampled, c))
        return reconstructed, latent_mu, latent_logvar

    def sample_eval(self, z: Optional[th.Tensor], c: th.Tensor) -> th.Tensor:
        with th.no_grad():
            return (
                self.decoder((z, c))
                if z is not None
                else self.decoder(
                    (th.randn((c.shape[0], self.lat_size), device=self.device), c)
                )
            )

    @property
    def device(self) -> th.device:
        return next(self.parameters()).device
