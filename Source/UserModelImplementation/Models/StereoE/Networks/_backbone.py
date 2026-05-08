# -*- coding: utf-8 -*-
from __future__ import annotations
from contextlib import nullcontext
from typing import List
import os

import torch
from torch import nn

from UserModelImplementation.Models.StereoD.Networks._backbone import (
    create_backbone as _create_dino_backbone,
    get_dino_layers_id,
)

try:
    from torchvision.models import convnext_large, ConvNeXt_Large_Weights
    from torchvision.models.feature_extraction import create_feature_extractor
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False
    convnext_large = None
    ConvNeXt_Large_Weights = None
    create_feature_extractor = None


class DinoBackboneAdapter(nn.Module):
    """Wrapper around dinov2/dinov3 backbones."""

    def __init__(self, backbone_name: str, model_name: str,
                 weights_path: str = None, freeze: bool = True) -> None:
        super().__init__()
        self.backbone = _create_dino_backbone(backbone_name, model_name, weights_path)
        self.layers = get_dino_layers_id(model_name)
        self.freeze = freeze

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        ctx = torch.no_grad() if self.freeze else nullcontext()
        with ctx:
            feats = self.backbone.get_intermediate_layers(
                x, n=self.layers, reshape=True, return_class_token=False, norm=False)
        return list(feats)


class DepthAnythingBackbone(nn.Module):
    """Approximation of DepthAnything style features with ConvNeXt fallback."""

    def __init__(self, weights_path: str = None, freeze: bool = True) -> None:
        super().__init__()
        self.freeze = freeze
        if _HAS_TORCHVISION:
            weights = ConvNeXt_Large_Weights.DEFAULT if weights_path is None else None
            backbone = convnext_large(weights=weights)
            if weights_path is not None and os.path.isfile(weights_path):
                state = torch.load(weights_path, map_location='cpu')
                backbone.load_state_dict(state, strict=False)
            return_nodes = {
                'features.1': 's1',
                'features.2': 's2',
                'features.3': 's3',
                'features.4': 's4',
            }
            self.return_order = ['s1', 's2', 's3', 's4']
            self.extractor = create_feature_extractor(backbone, return_nodes=return_nodes)
        else:
            # Minimal fallback CNN to keep pipeline running.
            self.return_order = ['s1', 's2', 's3', 's4']
            self.extractor = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            self.blocks = nn.ModuleList([
                nn.Sequential(nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.ReLU(inplace=True)),
                nn.Sequential(nn.Conv2d(128, 256, 3, padding=1, stride=2), nn.ReLU(inplace=True)),
                nn.Sequential(nn.Conv2d(256, 512, 3, padding=1, stride=2), nn.ReLU(inplace=True)),
            ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        ctx = torch.no_grad() if self.freeze else nullcontext()
        with ctx:
            if hasattr(self, "blocks"):
                feats = []
                x = self.extractor(x)
                feats.append(x)
                for block in self.blocks:
                    x = block(x)
                    feats.append(x)
                return feats
            feat_dict = self.extractor(x)
        return [feat_dict[k] for k in self.return_order]


def build_backbone(backbone: str, model_name: str = None,
                   weights_path: str = None, freeze: bool = True) -> nn.Module:
    backbone = backbone.lower()
    if backbone in {'dinov3', 'dinov2'}:
        model_name = model_name or ('dinov3_vith16plus' if backbone == 'dinov3' else 'dinov2_vitl14')
        return DinoBackboneAdapter(backbone, model_name, weights_path, freeze)
    if backbone in {'depthanything', 'depth-anything', 'depth'}:
        return DepthAnythingBackbone(weights_path=weights_path, freeze=freeze)
    raise ValueError(f'Unsupported backbone: {backbone}')

