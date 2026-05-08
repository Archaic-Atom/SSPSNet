# -*- coding: utf-8 -*-
from functools import partial

import os
import torch
from torch import nn
import torch.nn.functional as F


try:
    from ._backbone import get_dino_layers_id, create_backbone
    from ._cost_volume import build_cat_volume
    from ._disparity_regression import DispRegression
    from ._feature_matching import PSMNet
    from ._prompt import Prompt
    from ._upsampler import JBUStack
except ImportError:
    from _backbone import get_dino_layers_id, create_backbone
    from _cost_volume import build_cat_volume
    from _disparity_regression import DispRegression
    from _feature_matching import PSMNet
    from _prompt import Prompt
    from _upsampler import JBUStack


class StereoA(nn.Module):
    RECONSTRUCTION_CHANNELS = 1
    ID_FEAT = 0
    ROOT_PATH = '/data3/raozhibo/SAStereo/'
    WEIGHTS_PATH = 'Weights/dinov3/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth'

    def __init__(self, in_channles: int, start_disp: int = 1, disp_num: int = 192,
                 backbone_name: str = 'dinov3', pre_train_opt: bool = False,
                 confidence_level: float = 0.15) -> None:
        super().__init__()
        out_channels = [320, 640, 1280, 1280]
        self.start_disp, self.disp_num = start_disp, disp_num
        self.confidence_level = confidence_level
        self._h, self._w = None, None
        self.in_channles, self.pre_train_opt = in_channles, pre_train_opt
        self.backbone = create_backbone(backbone_name, 'dinov3_vith16plus',
                                        os.path.join(self.ROOT_PATH, self.WEIGHTS_PATH))
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels=1280, out_channels=out_channel,
                      kernel_size=1, stride=1, padding=0,) for out_channel in out_channels
        ])

        self.resize_layers = self._get_resize_layers(out_channels)
        self.fusion = nn.Conv2d(in_channels=220, out_channels=64,
                                kernel_size=1, stride=1, padding=0,)

        self.matching_module = PSMNet(128, start_disp, disp_num, True)
        self.prompt_module = Prompt(8)

        self.trainable_tail = nn.ModuleDict(dict(
            projects=self.projects,
            resize_layers=self.resize_layers,
            fusion=self.fusion,
            matching_module=self.matching_module,
            prompt_module=self.prompt_module
        ))

        # self.disp_regression = DispRegression([start_disp, start_disp + disp_num - 1])

    def _get_dinov2(self) -> nn.Module:
        # feature_extraction = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        feature_extraction = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14',
                                            source='local', pretrained=False)
        feature_extraction.forward = partial(
            feature_extraction.get_intermediate_layers,
            n=1, reshape=True, return_class_token=False, norm=False,)
        return feature_extraction

    def _get_resize_layers(self, out_channels) -> nn.Module:
        return nn.ModuleList([
            JBUStack(out_channels[0], out_channels[0] // 16),
            JBUStack(out_channels[1], out_channels[1] // 16),
            JBUStack(out_channels[2], out_channels[2] // 16),
            JBUStack(out_channels[3], out_channels[3] // 16),
        ])

    def _get_feature_extraction(self, backbone: str) -> nn.Module:
        if backbone in {"CNN"}:
            return None
        return self._get_dinov2()

    def _build_cost_volume_proc(self, left_img: torch.Tensor,
                                right_img: torch.Tensor) -> torch.Tensor:
        return build_cat_volume(left_img, right_img, self.start_disp, 48)

    def _up_sampling(self, cost: torch.Tensor, shape: list, decoder: nn.Module) -> torch.Tensor:
        return decoder(F.interpolate(cost, shape, mode='trilinear', align_corners=True))

    def _feature_extraction_module_proc(
            self, left_img: torch.Tensor, right_img: torch.Tensor) -> tuple:
        with torch.no_grad():
            left_img_list = self.backbone.get_intermediate_layers(
                left_img, n = get_dino_layers_id('dinov3_vith16plus'), reshape = True,
                return_class_token = False, norm = False)
            right_img_list = self.backbone.get_intermediate_layers(
                right_img, n = get_dino_layers_id('dinov3_vith16plus'), reshape = True,
                return_class_token = False, norm = False)

        left_img_layers, right_img_layers = [], []
        for i, (left_item, right_item) in enumerate(zip(left_img_list, right_img_list)):
            left_img_layers.append(self.resize_layers[i](self.projects[i](left_item), left_img))
            right_img_layers.append(self.resize_layers[i](self.projects[i](right_item), right_img))

        left_img_final = self.fusion(torch.cat(left_img_layers, dim=1))
        right_img_final = self.fusion(torch.cat(right_img_layers, dim=1))

        return left_img_final, right_img_final

    def _matching_module_proc(self, cost: torch.Tensor) -> torch.Tensor:
        return self.matching_module(cost,
                                    [self.disp_num, self._h, self._w],
                                    self.confidence_level)

    def _prompt_proc(self, disp_list, left_img, right_img, size):
        if self.training:
            disp, mask = disp_list[2], disp_list[6]
        else:
            disp, mask = disp_list[0], disp_list[1]

        return self.prompt_module(left_img, right_img, disp, mask, size)

    def _mask_fine_tune_proc(self, left_img: torch.Tensor, right_img: torch.Tensor) -> tuple:
        _, _, self._h, self._w = left_img.shape
        left_img, right_img = self._feature_extraction_module_proc(left_img, right_img)
        cost = self._build_cost_volume_proc(left_img, right_img)
        disp_list = self._matching_module_proc(cost)
        disp = self._prompt_proc(disp_list, left_img, right_img, (self._h, self._w))
        return disp_list + [disp]

    def _mask_pre_train_proc(self, left_img: torch.Tensor,
                             right_img: torch.Tensor) -> torch.Tensor:
        return list(self._feature_extraction_module_proc(left_img, right_img))

    def forward(self, left_img: torch.Tensor, right_img: torch.Tensor) -> torch.Tensor:
        if self.pre_train_opt:
            return self._mask_pre_train_proc(left_img, right_img)
        return self._mask_fine_tune_proc(left_img, right_img)


if __name__ == '__main__':
    model = StereoA(3).cuda()
    rand_tensor = torch.rand(1, 3, 640, 320).cuda()
    # model.eval()
    match_out = model(rand_tensor, rand_tensor)

    for item in match_out:
        print(item.shape)
