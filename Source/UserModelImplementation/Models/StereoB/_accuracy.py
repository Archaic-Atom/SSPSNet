# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn
import JackFramework as jf

try:
    from ._warp import Warp
except ImportError:
    from _warp import Warp


class Accuracy(object):
    DISP_DIM_LEN = 3
    ID_CHANNEL = 1
    EPS = 1e-6

    def __init__(self, args: object) -> None:
        super().__init__()
        self.__arg = args
        self._warp = Warp()
        self._cos = nn.CosineSimilarity(dim=self.ID_CHANNEL, eps=self.EPS)

    def matching_accuracy(self, disp_list: list, disp_label: torch.Tensor,
                          id_error_px: int = 1, invalid_value: int = 0) -> list:
        res = []
        for _, disp in enumerate(disp_list):
            if len(disp.shape) == self.DISP_DIM_LEN:
                acc, mae = jf.acc.SMAccuracy.d_1(disp, disp_label, invalid_value)
                res.extend((acc[id_error_px], mae))
        return res

    def feature_alignment_accuracy(self, left_feat: torch.Tensor, right_feat: torch.Tensor,
                                   disp_label: torch.Tensor, mask_disp: torch.Tensor) -> list:
        if len(disp_label.shape) == self.DISP_DIM_LEN:
            disp_label = disp_label.unsqueeze(self.ID_CHANNEL)

        _, _, h, w = disp_label.shape
        left_feat = F.interpolate(left_feat, [h, w], mode='bilinear', align_corners=False)
        right_feat = F.interpolate(right_feat, [h, w], mode='bilinear', align_corners=False)

        warped_right_img = self._warp(left_feat, right_feat, disp_label)
        mask_occ = self._warp(torch.ones_like(disp_label), torch.ones_like(disp_label), disp_label)
        mask = mask_disp.float() * mask_occ

        return [torch.mean(torch.sum(torch.abs(left_feat - warped_right_img),
                dim=self.ID_CHANNEL, keepdim=True) * mask),
                torch.mean(self._cos(left_feat, warped_right_img) * mask)]
