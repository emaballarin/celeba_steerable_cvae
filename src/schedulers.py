#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List

import torch


def warmed_up_linneal(
    optim: torch.optim.Optimizer,
    init_lr: float,
    steady_lr: float,
    final_lr: float,
    warmup_epochs: int,
    steady_epochs: int,
    anneal_epochs: int,
):
    # Prepare optim
    for grp in optim.param_groups:
        grp["lr"] = steady_lr

    milestones: List[int] = [
        max(3, warmup_epochs),
        max(3, warmup_epochs) + max(1, steady_epochs),
    ]

    # Schedulers
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optim,
        start_factor=init_lr / steady_lr,
        end_factor=1.0,
        total_iters=milestones[0],
        last_epoch=-1,
    )
    steady_scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer=optim,
        factor=1.0,
        total_iters=milestones[1],
        last_epoch=-1,
    )
    anneal_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optim,
        start_factor=1.0,
        end_factor=final_lr / steady_lr,
        total_iters=max(3, anneal_epochs),
        last_epoch=-1,
    )

    # Prepare scheduler
    sched = torch.optim.lr_scheduler.SequentialLR(
        optim,
        schedulers=[warmup_scheduler, steady_scheduler, anneal_scheduler],
        milestones=milestones,
        last_epoch=-1,
    )

    # Return
    return optim, sched
