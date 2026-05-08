# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F

from UserModelImplementation.Models.StereoA._warp import Warp


class Loss(object):
    """Loss utilities for StereoE."""
    DISP_DIM_LEN = 3
    ID_CHANNEL = 1
    PROB_WEIGHT = 0.02

    def __init__(self, args: object) -> None:
        super().__init__()
        self.__arg = args
        self._warp = Warp()

    def _alignment_loss(self, left_feat: torch.Tensor, right_feat: torch.Tensor,
                        disp_label: torch.Tensor, mask_disp: torch.Tensor) -> torch.Tensor:
        warped_right_img = self._warp(left_feat, right_feat, disp_label)
        mask_occ = self._warp(torch.ones_like(disp_label), torch.ones_like(disp_label), disp_label)
        mask = mask_disp.float() * mask_occ
        return torch.mean(torch.sum(torch.abs(left_feat - warped_right_img),
                                    dim=self.ID_CHANNEL, keepdim=True) * mask)

    @staticmethod
    def _disp2distribute(start_disp: int, disp_gt: torch.Tensor,
                         max_disp: int, b: float = 2.0) -> torch.Tensor:
        disp_gt = disp_gt.unsqueeze(1)
        disp_range = torch.arange(start_disp, start_disp + max_disp).view(1, -1, 1, 1).float().to(disp_gt.device)
        gt_distribute = torch.exp(-torch.abs(disp_range - disp_gt) / b)
        gt_distribute = gt_distribute / (torch.sum(gt_distribute, dim=1, keepdim=True) + 1e-8)
        return gt_distribute

    def _probability_loss(self, prob_lr: torch.Tensor,
                          disp_label: torch.Tensor, mask_disp: torch.Tensor) -> torch.Tensor:
        B, _, H4, W4 = prob_lr.shape
        disp_lr = F.interpolate(disp_label.unsqueeze(1), size=(H4, W4), mode='area').squeeze(1)
        mask_lr = F.interpolate(mask_disp.unsqueeze(1).float(), size=(H4, W4), mode='nearest').squeeze(1).bool()

        gt_distribute = self._disp2distribute(
            self.__arg.start_disp, disp_lr, self.__arg.disp_num, b=2.0)
        pred = torch.log(prob_lr + 1e-8)
        ce_loss = torch.sum(-gt_distribute * pred, dim=1)
        ce_loss = ce_loss[mask_lr].mean() if mask_lr.any() else ce_loss.mean()
        return ce_loss

    def matching_loss(self, output_data: list, disp_label: torch.Tensor,
                      mask_disp: torch.Tensor) -> list:
        """Compute full/coarse/prompt consistency losses."""
        match_out = output_data[0]
        disp_full = match_out["disp_full"]
        disp_coarse = match_out["disp_coarse"]
        disp_aux = match_out.get("disp_aux")
        prob_lr = match_out.get("prob_prompt")

        loss_full = F.smooth_l1_loss(disp_full[mask_disp], disp_label[mask_disp])
        loss_coarse = F.smooth_l1_loss(disp_coarse[mask_disp], disp_label[mask_disp])
        total_loss = 0.7 * loss_full + 0.3 * loss_coarse

        aux_loss = None
        if disp_aux is not None:
            aux_loss = F.smooth_l1_loss(disp_aux[mask_disp], disp_label[mask_disp])
            total_loss = total_loss + 0.2 * aux_loss

        prob_loss = None
        if prob_lr is not None:
            prob_loss = self._probability_loss(prob_lr, disp_label, mask_disp)
            total_loss = total_loss + self.PROB_WEIGHT * prob_loss

        res = [total_loss, loss_full, loss_coarse]
        if aux_loss is not None:
            res.append(aux_loss)
        if prob_loss is not None:
            res.append(prob_loss)
        return res

    def feature_alignment_loss(self, left_feat: torch.Tensor, right_feat: torch.Tensor,
                               disp_label: torch.Tensor, mask_disp: torch.Tensor) -> list:
        if len(disp_label.shape) == self.DISP_DIM_LEN:
            disp_label = disp_label.unsqueeze(self.ID_CHANNEL)

        _, _, h, w = disp_label.shape
        left_feat = F.interpolate(left_feat, [h, w], mode='bilinear', align_corners=False)
        right_feat = F.interpolate(right_feat, [h, w], mode='bilinear', align_corners=False)

        alignment_loss = self._alignment_loss(left_feat, right_feat, disp_label, mask_disp)
        return [alignment_loss, alignment_loss]

