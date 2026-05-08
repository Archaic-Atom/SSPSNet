# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

from UserModelImplementation.Models.StereoA.Networks import DispRegression


class _Depthwise3DBlock(nn.Module):
    """Lightweight residual 3D convolution block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depth = nn.Conv3d(channels, channels, kernel_size=3, padding=1,
                               groups=channels, bias=False)
        self.point = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depth(x)
        y = self.point(y)
        y = self.norm(y)
        return self.act(x + y)


class LiteMatchingHead(nn.Module):
    """Shallow 3D aggregation head that produces logits + auxiliary branch."""

    def __init__(self, in_channels: int, start_disp: int, disp_num: int,
                 base_channels: int = 64, num_blocks: int = 3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([_Depthwise3DBlock(base_channels) for _ in range(num_blocks)])
        self.classifier = nn.Conv3d(base_channels, 1, kernel_size=3, padding=1, bias=True)
        self.aux_classifier = nn.Conv3d(base_channels, 1, kernel_size=3, padding=1, bias=True)
        self.regression = DispRegression([start_disp, start_disp + disp_num - 1])

    def forward(self, cost_volume: torch.Tensor) -> dict:
        x = self.stem(cost_volume)
        aux_logits = self.aux_classifier(x).squeeze(1)

        for block in self.blocks:
            x = block(x)

        logits = self.classifier(x).squeeze(1)
        prob = F.softmax(logits, dim=1)
        aux_prob = F.softmax(aux_logits, dim=1)
        disp = self.regression(logits)
        disp_aux = self.regression(aux_logits)
        confidence = prob.max(dim=1, keepdim=True)[0]

        return {
            "logits": logits,
            "prob": prob,
            "disp": disp,
            "aux_logits": aux_logits,
            "aux_prob": aux_prob,
            "aux_disp": disp_aux,
            "confidence": confidence,
        }

