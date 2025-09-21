# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, Optional, Dict

import torch
import torch.nn.functional as F
import JackFramework as jf

try:
    from ._warp import Warp
    from .losses_prob import ProbVolumeLoss
except ImportError:
    from _warp import Warp
    from losses_prob import ProbVolumeLoss


class Loss(object):
    DISP_DIM_LEN, GROUPED_NUM = 3, 8
    ID_CHANNEL = 1

    def __init__(self, args: object) -> None:
        super().__init__()
        self.__arg = args
        self._warp = Warp()
        self.prob_loss = ProbVolumeLoss(radius=2, w_prob=0.02, w_prob1=0.02, w_entropy=0.01)

    def matching_accuracy(self, disp_list: list, disp_label: torch.Tensor,
                          id_error_px: int = 1, invalid_value: int = 0) -> list:
        res = []
        for _, disp in enumerate(disp_list):
            if len(disp.shape) == self.DISP_DIM_LEN:
                acc, mae = jf.acc.SMAccuracy.d_1(disp, disp_label, invalid_value)
                res.extend((acc[id_error_px], mae))
        return res

    def _alignment_loss(self, left_feat: torch.Tensor, right_feat: torch.Tensor,
                        disp_label: torch.Tensor, mask_disp: torch.Tensor) -> None:
        warped_right_img = self._warp(left_feat, right_feat, disp_label)
        mask_occ = self._warp(torch.ones_like(disp_label), torch.ones_like(disp_label), disp_label)
        mask = mask_disp.float() * mask_occ
        return torch.mean(torch.sum(torch.abs(left_feat - warped_right_img),
                                    dim=self.ID_CHANNEL, keepdim=True) * mask)

    @staticmethod
    def _disp2distribute(start_disp, disp_gt, max_disp, b=2):
        disp_gt = disp_gt.unsqueeze(1)
        disp_range = torch.arange(start_disp, start_disp + max_disp).view(1, -1, 1, 1).float().cuda()
        gt_distribute = torch.exp(-torch.abs(disp_range - disp_gt) / b)
        gt_distribute = gt_distribute / (torch.sum(gt_distribute, dim=1, keepdim=True) + 1e-8)
        return gt_distribute

    @staticmethod
    def _celoss(start_disp, disp_gt, max_disp, gt_distribute, pred_distribute):
        mask = (disp_gt > start_disp) & (disp_gt < start_disp + max_disp)

        pred_distribute = torch.log(pred_distribute + 1e-8)
        ce_loss = torch.sum(-gt_distribute * pred_distribute, dim=1)
        ce_loss = torch.mean(ce_loss[mask])
        return ce_loss

    def matching_loss(
            self, disp_list: list, disp_label: torch.Tensor, mask_disp: torch.Tensor) -> torch.Tensor:
        # args = self.__arg

        # gt_distribute = self._disp2distribute(args.start_disp, disp_label, args.disp_num, b=2)
        res = []
        match_out_dict = disp_list[0]

        loss_dict = self.prob_loss(
            prob=match_out_dict["prob"],
            prob1=match_out_dict["aux"]["prob1"],
            disp_gt_full=disp_label,                  # [B,1,H,W], FULL-RES 像素
            mask_full=mask_disp,
            orig_hw=(disp_label.shape[-2], disp_label.shape[-1]),
            H4W4=(match_out_dict["prob"].shape[-2], match_out_dict["prob"].shape[-1]),
            D=match_out_dict["prob"].shape[1],
            band_dmin=match_out_dict["band"][0],                        # ✅ 推荐
            band_offset_pixels=match_out_dict["band_offset_px"]         # 或者传 band_offset_pixels（full-res 像素）
        )

        loss_l1 = F.smooth_l1_loss(match_out_dict["disp_full"][mask_disp.unsqueeze(1)],
                                   disp_label.unsqueeze(1)[mask_disp.unsqueeze(1)])

        loss = loss_dict["loss"] + loss_l1
        res.append(loss)
        res.append(loss_dict["loss"])
        res.append(loss_l1)
        return res

    def feature_alignment_loss(self, left_feat: torch.Tensor, right_feat: torch.Tensor,
                               disp_label: torch.Tensor, mask_disp: torch.Tensor) -> list:
        if len(disp_label.shape) == self.DISP_DIM_LEN:
            disp_label = disp_label.unsqueeze(self.ID_CHANNEL)

        _, _, h, w = disp_label.shape
        left_feat = F.interpolate(left_feat, [h, w], mode = 'bilinear', align_corners = False)
        right_feat = F.interpolate(right_feat, [h, w], mode = 'bilinear', align_corners = False)

        alignment_loss = self._alignment_loss(left_feat, right_feat, disp_label, mask_disp)

        return [alignment_loss, alignment_loss]
