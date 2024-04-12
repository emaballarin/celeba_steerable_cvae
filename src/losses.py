#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import Tensor


@torch.jit.script
def pixelwise_bce_sum(lhs: Tensor, rhs: Tensor) -> Tensor:
    return F.binary_cross_entropy(lhs, rhs, reduction="sum")


@torch.jit.script
def beta_gaussian_kldiv(mu: Tensor, sigma: Tensor, beta: float = 1.0) -> Tensor:
    kldiv = 0.5 * (torch.pow(mu, 2) + torch.exp(sigma) - sigma - 1).sum()
    return beta * kldiv


@torch.jit.script
def beta_reco_bce(
    input_reco: Tensor,
    input_orig: Tensor,
    mu: Tensor,
    sigma: Tensor,
    beta: float = 1.0,
):
    kldiv = beta_gaussian_kldiv(mu, sigma, beta)
    pwbce = pixelwise_bce_sum(input_reco, input_orig)
    return pwbce + kldiv


@torch.jit.script
def beta_reco_bce_splitout(
    input_reco: Tensor,
    input_orig: Tensor,
    mu: Tensor,
    sigma: Tensor,
    beta: float = 1.0,
):
    kldiv = beta_gaussian_kldiv(mu, sigma, beta)
    pwbce = pixelwise_bce_sum(input_reco, input_orig)
    return pwbce + kldiv, pwbce, kldiv


@torch.jit.script
def var_of_lap(img: torch.Tensor) -> torch.Tensor:
    lap_kernel = (
        torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=img.device
        )
        .expand(img.shape[-3], 3, 3)
        .unsqueeze(1)
    )
    return (
        torch.nn.functional.conv2d(img, lap_kernel, groups=img.shape[-3])
        .var(dim=(-2, -1))
        .sum(-1)
    )
