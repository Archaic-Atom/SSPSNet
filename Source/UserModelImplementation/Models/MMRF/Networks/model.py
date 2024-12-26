# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

try:
    from ._feature_extraction import Feature
    from ._cost_volume import build_gwc_volume
    from ._dpn import DPN
    from ._mmrf import NMRF
except ImportError:
    from _feature_extraction import Feature
    from _cost_volume import build_gwc_volume
    from _dpn import DPN
    from _mmrf import NMRF


class FANet(nn.Module):
    RECONSTRUCTION_CHANNELS = 1
    ID_FEAT = 0
    GROUP_NUM = 8
    IMAGE_SCALING_RATIO = 14

    def __init__(self, in_channles: int, start_disp: int, disp_num: int,
                 backbone_name: str, pre_train_opt: bool) -> None:
        super().__init__()
        assert disp_num % self.IMAGE_SCALING_RATIO == 0
        self.start_disp, self.disp_num = start_disp, disp_num
        self._h, self._w = None, None
        self.in_channles, self.backbone, self.pre_train_opt =\
            in_channles, backbone_name, pre_train_opt
        self.feature_extraction = self._get_feature_extraction()
        self.matching_module = DPN(cost_group=8, feat_dim=1024)
        self.mmrf = NMRF(1024, self.IMAGE_SCALING_RATIO)
        self.disp_regression = None

    def _get_dinov2(self) -> nn.Module:
        return torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', source='local')

    def _get_feature_extraction(self) -> nn.Module:
        if self.backbone in {"CNN"}:
            return Feature()
        return self._get_dinov2()

    def _build_cost_volume_proc(self, left_img: torch.Tensor, right_img: torch.Tensor,
                                start_disp: int, disp_num: int,
                                group_num: int = 8) -> torch.Tensor:
        return build_gwc_volume(left_img, right_img, start_disp, disp_num, group_num)

    def _up_sampling(self, cost: torch.Tensor, shape: list, decoder: nn.Module) -> torch.Tensor:
        return decoder(F.interpolate(cost, shape, mode='trilinear', align_corners=True))

    def _feature_extraction_module_proc(
            self, left_img: torch.Tensor, right_img: torch.Tensor) -> tuple:
        left_img = self.feature_extraction.get_intermediate_layers(
            left_img, n=1, reshape=True, return_class_token=False, norm=False)[self.ID_FEAT]
        right_img = self.feature_extraction.get_intermediate_layers(
            right_img, n=1, reshape=True, return_class_token=False, norm=False)[self.ID_FEAT]
        return left_img, right_img

    def _matching_module_proc(self, cost: torch.Tensor) -> torch.Tensor:
        return cost

    def _mask_fine_tune_proc(self, left_img: torch.Tensor,
                             right_img: torch.Tensor) -> tuple:
        _, _, self._h, self._w = left_img.shape
        left_img, right_img = self._feature_extraction_module_proc(left_img, right_img)
        cost = self._build_cost_volume_proc(
            left_img, right_img, self.start_disp, int(self.disp_num / self.IMAGE_SCALING_RATIO))
        _, prob, _, labels = self.matching_module(cost, left_img)
        disp, coarse_disp, mask = self.mmrf(left_img, right_img, labels)
        print('model', prob.shape)
        prob = F.interpolate(prob.unsqueeze(1), [self.disp_num, self._h, self._w], mode='trilinear', align_corners=True)
        prob = prob.squeeze(1)
        print('model', prob.shape)

        return disp, coarse_disp, mask, prob

    def _mask_pre_train_proc(self, left_img: torch.Tensor,
                             right_img: torch.Tensor) -> torch.Tensor:
        return list(self._feature_extraction_module_proc(left_img, right_img))

    def forward(self, left_img: torch.Tensor, right_img: torch.Tensor) -> torch.Tensor:
        if self.pre_train_opt:
            return self._mask_pre_train_proc(left_img, right_img)
        return self._mask_fine_tune_proc(left_img, right_img)
