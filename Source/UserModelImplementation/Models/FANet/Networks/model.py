# -*- coding: utf-8 -*-
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F


try:
    from ._feature_extraction import Feature
    from ._cost_volume import build_gwc_volume, build_cat_volume
    from ._disparity_regression import DispRegression
    from .Transformer import build_vit_matching_module
except ImportError:
    from _feature_extraction import Feature
    from _cost_volume import build_gwc_volume, build_cat_volume
    from _disparity_regression import DispRegression
    from Transformer import build_vit_matching_module


class FANet(nn.Module):
    RECONSTRUCTION_CHANNELS = 1
    ID_FEAT = 0
    GROUP_NUM = 8

    def __init__(self, in_channles: int, start_disp: int, disp_num: int,
                 backbone: str, pre_train_opt: bool) -> None:
        super().__init__()
        self.start_disp, self.disp_num = start_disp, disp_num
        self._h, self._w = None, None
        self.in_channles = in_channles
        self.backbone = backbone
        self.pre_train_opt = pre_train_opt
        self.feature_extraction = self._get_feature_extraction()
        self.matching_module = build_vit_matching_module(776)
        self.deonv_0, self.deonv_1, self.deonv_2 = self._get_decode_module()
        self.disp_regression = DispRegression([start_disp, start_disp + disp_num - 1])

    def _get_dinov2(self) -> nn.Module:
        feature_extraction = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        feature_extraction.forward = partial(
            feature_extraction.get_intermediate_layers,
            n=1,
            reshape=True,
            return_class_token=False,
            norm=False,
        )
        return feature_extraction

    def _get_feature_extraction(self) -> nn.Module:
        if self.backbone in {"CNN"}:
            return Feature()
        return self._get_dinov2()

    def _get_decode_module(self) -> nn.Module:
        deonv_1 = nn.Sequential(nn.Conv3d(192, 64, kernel_size=3, padding=1, stride=1, bias=False),
                                nn.BatchNorm3d(64),
                                nn.ReLU(inplace=True))
        deonv_2 = nn.Sequential(nn.Conv3d(64, 8, kernel_size=3, padding=1, stride=1, bias=False),
                                nn.BatchNorm3d(8),
                                nn.ReLU(inplace=True))
        deonv_3 = nn.Sequential(nn.Conv3d(8, 1, kernel_size=3, padding=1, stride=1, bias=False))
        return deonv_1, deonv_2, deonv_3

    def _build_cost_volume_proc(self, left_img: torch.Tensor,
                                right_img: torch.Tensor) -> torch.Tensor:
        return torch.cat((
            build_gwc_volume(left_img, right_img, self.start_disp, 14, self.GROUP_NUM),
            build_cat_volume(left_img, right_img, self.start_disp, 14)),
            dim=1)

    def _up_sampling(self, cost: torch.Tensor, shape: list, decoder: nn.Module) -> torch.Tensor:
        cost = F.interpolate(cost, shape, mode='trilinear', align_corners=True)
        return decoder(cost)

    def _feature_extraction_module_proc(self, left_img: torch.Tensor, right_img: torch.Tensor) -> tuple:
        left_img = self.feature_extraction.get_intermediate_layers(
            left_img, n=1, reshape=True, return_class_token=False, norm=False)[self.ID_FEAT]
        right_img = self.feature_extraction.get_intermediate_layers(
            right_img, n=1, reshape=True, return_class_token=False, norm=False)[self.ID_FEAT]
        return left_img, right_img

    def _matching_module_proc(self, cost: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = cost.shape
        cost = cost.reshape(b, c, -1, w)
        cost = self.matching_module.get_intermediate_layers(
            cost, n=1, reshape=True, return_class_token=False, norm=False)[self.ID_FEAT]
        cost = cost.reshape(b, -1, d, h, w)
        cost = self._up_sampling(cost, [d * 2, h * 2, w * 2], self.deonv_0)
        cost = self._up_sampling(cost, [d * 4, h * 4, w * 4], self.deonv_1)
        cost = self._up_sampling(cost, [self.disp_num, self._h, self._w], self.deonv_2)
        return cost

    def _mask_fine_tune_proc(self, left_img: torch.Tensor, right_img: torch.Tensor,
                             flow_init: torch.Tensor = None) -> tuple:
        _, _, self._h, self._w = left_img.shape
        left_img, right_img = self._feature_extraction_module_proc(left_img, right_img)
        cost = self._build_cost_volume_proc(left_img, right_img)
        cost = self._matching_module_proc(cost)
        disp = self.disp_regression(-torch.squeeze(cost, 1))
        return disp

    def _mask_pre_train_proc(self, left_img: torch.Tensor,
                             right_img: torch.Tensor) -> torch.Tensor:
        return [
            self.feature_extraction.get_intermediate_layers(
                left_img, n=1, reshape=True, return_class_token=False, norm=False)[self.ID_FEAT],
            self.feature_extraction.get_intermediate_layers(
                right_img, n=1, reshape=True, return_class_token=False, norm=False)[self.ID_FEAT]
        ]

    def forward(self, left_img: torch.Tensor, right_img: torch.Tensor,
                random_sample_list: torch.Tensor = None, flow_init=None) -> torch.Tensor:
        if self.pre_train_opt:
            return self._mask_pre_train_proc(left_img, right_img)
        return self._mask_fine_tune_proc(left_img, right_img, flow_init)
