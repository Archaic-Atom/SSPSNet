# -*- coding: utf-8 -*-
import torch.nn as nn
from .model import MatchingModule


def build_hourglass_module(in_planes: int) -> nn.Module:
    return MatchingModule(in_planes)
