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
