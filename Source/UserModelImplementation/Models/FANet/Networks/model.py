# -*- coding: utf-8 -*-
import torch
from torch import nn
from .feature_extraction import Feature


class FANet(nn.Module):
    RECONSTRUCTION_CHANNELS = 1

    def __init__(self, in_channles: int, start_disp: int, disp_num: int, pre_train_opt: bool) -> None:
        super().__init__()
        self.start_disp, self.disp_num = start_disp, disp_num
        self.in_channles = in_channles
        self.pre_train_opt = pre_train_opt
        self.feature_extraction = Feature()

    def _mask_pre_train_proc(self, left_img: torch.Tensor, right_img: torch.Tensor) -> torch.Tensor:
        left_img = self.feature_extraction(left_img)
        right_img = self.feature_extraction(right_img)

        return [left_img, right_img]

    def _mask_fine_tune_proc(self, left_img: torch.Tensor, right_img: torch.Tensor,
                             flow_init: torch.Tensor) -> tuple:
        return 0

    def forward(self, left_img: torch.Tensor, right_img: torch.Tensor,
                random_sample_list: torch.Tensor = None, flow_init=None) -> torch.Tensor:
        if self.pre_train_opt:
            return self._mask_pre_train_proc(left_img, right_img)
        return self._mask_fine_tune_proc(left_img, right_img, flow_init)
