#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import math
from copy import deepcopy

import torch
import torch.nn as nn

__all__ = ["ModelEMA", "is_parallel"]


def is_parallel(model):
    """check if model is in parallel mode."""
    return isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel))


class ModelEMA:
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is much simpler than other implementations.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay rate.
            updates (int): counter of EMA updates.
        """
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        self.updates = updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = (
                model.module.state_dict() if is_parallel(model) else model.state_dict()
            )  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()

    def state_dict(self):
        # 返回 EMA 模型的 state_dict
        return self.ema.state_dict()

    def load_state_dict(self, state_dict):
        # 加载 EMA 模型的 state_dict
        self.ema.load_state_dict(state_dict)