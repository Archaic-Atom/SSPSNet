# -*- coding: utf-8 -*-
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F


try:
    from ._cost_volume import build_cat_volume
    from ._disparity_regression import DispRegression
    from ._feature_matching import PSMNet
except ImportError:
    from _cost_volume import build_cat_volume
    from _disparity_regression import DispRegression
    from _feature_matching import PSMNet


class StereoA(nn.Module):
    RECONSTRUCTION_CHANNELS = 1
    ID_FEAT = 0
    GROUP_NUM = 512

    def __init__(self, in_channles: int, start_disp: int, disp_num: int,
                 backbone: str, pre_train_opt: bool) -> None:
        super().__init__()
        self.start_disp, self.disp_num = start_disp, disp_num
        self._h, self._w = None, None
        self.in_channles, self.backbone, self.pre_train_opt = in_channles, backbone, pre_train_opt
        self.pretrained = self._get_feature_extraction()
        out_channels = [256, 512, 1024, 1024]
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels=1024, out_channels=out_channel,
                      kernel_size=1, stride=1, padding=0,) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=out_channels[0], out_channels=out_channels[0] // 16,
                               kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(in_channels=out_channels[1], out_channels=out_channels[1] // 16,
                               kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(in_channels=out_channels[2], out_channels=out_channels[2] // 16,
                               kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(in_channels=out_channels[3], out_channels=out_channels[3] // 16,
                               kernel_size=4, stride=4, padding=0),
        ])

        self.project = nn.Conv2d(in_channels=176, out_channels=64,
                                 kernel_size=1, stride=1, padding=0,)

        self.matching_module = PSMNet(128, start_disp, disp_num, True)
        self.disp_regression = DispRegression([start_disp, start_disp + disp_num - 1])

    def _get_dinov2(self) -> nn.Module:
        # feature_extraction = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        feature_extraction = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14',
                                            source='local', pretrained=False)
        feature_extraction.forward = partial(
            feature_extraction.get_intermediate_layers,
            n=1, reshape=True, return_class_token=False, norm=False,)
        return feature_extraction

    def _get_feature_extraction(self) -> nn.Module:
        if self.backbone in {"CNN"}:
            return None
        return self._get_dinov2()

    def _get_decode_module(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv3d(384, 1, kernel_size=7, padding='same', stride=1, bias=False))

    def _get_decode_module(self) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(1024, 1, kernel_size=7, padding='same', stride=1, bias=False))

    def _build_cost_volume_proc(self, left_img: torch.Tensor,
                                right_img: torch.Tensor) -> torch.Tensor:
        return build_cat_volume(left_img, right_img, self.start_disp, 56)

    def _up_sampling(self, cost: torch.Tensor, shape: list, decoder: nn.Module) -> torch.Tensor:
        return decoder(F.interpolate(cost, shape, mode='trilinear', align_corners=True))

    def _feature_extraction_module_proc(
            self, left_img: torch.Tensor, right_img: torch.Tensor) -> tuple:
        left_img = self.pretrained.get_intermediate_layers(
            left_img, n = [4, 11, 17, 23], reshape = True, return_class_token = False, norm = False)
        right_img = self.pretrained.get_intermediate_layers(
            right_img, n = [4, 11, 17, 23], reshape = True, return_class_token = False, norm = False)

        left_img_layers, right_img_layers = [], []
        for i, (left_item, right_item) in enumerate(zip(left_img, right_img)):
            left_img_layers.append(self.resize_layers[i](self.projects[i](left_item)))
            right_img_layers.append(self.resize_layers[i](self.projects[i](right_item)))

        left_img_final = self.project(torch.cat(left_img_layers, dim=1))
        right_img_final = self.project(torch.cat(right_img_layers, dim=1))

        return left_img_final, right_img_final

    def _matching_module_proc(self, cost: torch.Tensor) -> torch.Tensor:
        return self.matching_module(cost, [self.disp_num, self._h, self._w])

    def _mask_fine_tune_proc(self, left_img: torch.Tensor, right_img: torch.Tensor,
                             flow_init: torch.Tensor = None) -> tuple:
        _, _, self._h, self._w = left_img.shape
        left_img, right_img = self._feature_extraction_module_proc(left_img, right_img)
        cost = self._build_cost_volume_proc(left_img, right_img)
        disp_list = self._matching_module_proc(cost)
        return disp_list

    def _mask_pre_train_proc(self, left_img: torch.Tensor,
                             right_img: torch.Tensor) -> torch.Tensor:
        return list(self._feature_extraction_module_proc(left_img, right_img))

    def forward(self, left_img: torch.Tensor, right_img: torch.Tensor,
                random_sample_list: torch.Tensor = None, flow_init=None) -> torch.Tensor:
        if self.pre_train_opt:
            return self._mask_pre_train_proc(left_img, right_img)
        return self._mask_fine_tune_proc(left_img, right_img, flow_init)
