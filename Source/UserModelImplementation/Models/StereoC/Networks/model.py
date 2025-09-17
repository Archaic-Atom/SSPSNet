# -*- coding: utf-8 -*-
from functools import partial

import os
import torch
from torch import nn
import torch.nn.functional as F


try:
    from ._backbone import get_dino_layers_id, create_backbone
    from ._upsampler import JBUStack
except ImportError:
    from _backbone import get_dino_layers_id, create_backbone
    from _upsampler import JBUStack


class StereoA(nn.Module):
    ROOT_PATH = '/data3/raozhibo/SAStereo/'
    WEIGHTS_PATH = 'Weights/dinov3/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth'

    def __init__(self, in_channles: int, start_disp: int, disp_num: int,
                 backbone_name: str = 'dinov3', confidence_level: float = 0.15) -> None:
        super().__init__()
        self.start_disp, self.disp_num = start_disp, disp_num
        self.in_channles = in_channles
        self.confidence_level = confidence_level
        self.backbone = create_backbone(backbone_name, 'dinov3_vith16plus',
                                        os.path.join(self.ROOT_PATH, self.WEIGHTS_PATH))
        self.resize_layers = self._get_resize_layers(1280)

    def _get_resize_layers(self, in_channles: int) -> nn.Module:
        return nn.ModuleList([
            JBUStack(feat_dim=in_channles, out_dim=256, mid_dim=256, radius=3)
            for i in range(len(get_dino_layers_id('dinov3_vith16plus')))
        ])

    def _feature_extraction_module_proc(
            self, left_img: torch.Tensor, right_img: torch.Tensor) -> tuple:

        with torch.no_grad():
            left_feat_list = self.backbone.get_intermediate_layers(
                left_img, n = get_dino_layers_id('dinov3_vith16plus'), reshape = True,
                return_class_token = False, norm = False)
            right_feat_list = self.backbone.get_intermediate_layers(
                right_img, n = get_dino_layers_id('dinov3_vith16plus'), reshape = True,
                return_class_token = False, norm = False)

        left_feat_list = [j(f, left_img) for j, f in zip(self.resize_layers, left_feat_list)]  # 每个→[B,256,H/4,W/4]
        right_feat_list = [j(f, right_img) for j, f in zip(self.resize_layers, right_feat_list)]

        for i in left_feat_list:
            print(i.shape)

        return left_feat_list, right_feat_list

    def _mask_fine_tune_proc(self, left_img: torch.Tensor, right_img: torch.Tensor) -> tuple:
        left_img, right_img = self._feature_extraction_module_proc(left_img, right_img)

    def forward(self, left_img: torch.Tensor, right_img: torch.Tensor) -> torch.Tensor:
        left_img, right_img = self._feature_extraction_module_proc(left_img, right_img)
        return None


if __name__ == '__main__':
    model = StereoA(3, 1, 192).cuda()
    rand_tensor = torch.rand(1, 3, 224, 224).cuda()
    model(rand_tensor, rand_tensor)
