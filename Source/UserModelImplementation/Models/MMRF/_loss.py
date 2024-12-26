# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import JackFramework as jf

try:
    from ._warp import Warp
    from .Networks import build_gwc_volume
    from .Networks import DispRegression
except ImportError:
    from _warp import Warp
    from Networks import build_gwc_volume
    from Networks import DispRegression


class Loss(object):
    DISP_DIM_LEN, GROUPED_NUM = 3, 8
    ID_CHANNEL = 1

    def __init__(self, args: object) -> None:
        super().__init__()
        self.__arg = args
        self._warp = Warp()
        self._disp_regression = DispRegression(
            [args.start_disp, args.start_disp + args.disp_num - 1])

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

    def _feature_matching_loss(self, left_feat: torch.Tensor, right_feat: torch.Tensor,
                               disp_label: torch.Tensor, mask_disp: torch.Tensor) -> None:
        args = self.__arg
        cost = build_gwc_volume(left_feat, right_feat, args.start_disp, args.disp_num, 8)
        cost = torch.mean(cost, dim=self.ID_CHANNEL, keepdim=False)
        disp = self._disp_regression(-cost)
        return F.smooth_l1_loss(disp[mask_disp.unsqueeze(1)], disp_label[mask_disp.unsqueeze(1)])

    def matching_loss(self, disp_list: list, disp_label: torch.Tensor,
                      mask_disp: torch.Tensor) -> torch.Tensor:
        res = None
        for i, disp in enumerate(disp_list):
            if i == 0:
                res = F.smooth_l1_loss(disp[mask_disp], disp_label[mask_disp])
            else:
                res += F.smooth_l1_loss(disp[mask_disp], disp_label[mask_disp])
        return res

    def loss_coarse(self, disp_pred, logits_pred, disp_gt, mask_disp: torch.Tensor):
        prob = F.softmax(logits_pred, dim=-1)
        print(disp_gt.shape, prob.shape, disp_pred.shape)
        error = F.smooth_l1_loss(disp_pred, disp_gt, reduction='none').unsqueeze(-1)
        print('error', error.shape)

        if torch.any(mask_disp):
            loss = torch.sum((prob * error)[-1], dim=-1, keepdim=False)[mask_disp].mean()
        else:  # dummy loss
            loss = F.smooth_l1_loss(disp_pred, disp_pred.detach(), reduction='mean') + \
                F.smooth_l1_loss(logits_pred, logits_pred.detach(), reduction='mean')
        return loss

    def prob_loss(self, prob, disp_gt, mask_disp: torch.Tensor):
        print('prob_loss', prob.shape)
        disp = self._disp_regression(prob).squeeze(1)
        res = F.smooth_l1_loss(disp[mask_disp], disp_gt[mask_disp])
        return res

    def feature_alignment_loss(self, left_feat: torch.Tensor, right_feat: torch.Tensor,
                               disp_label: torch.Tensor, mask_disp: torch.Tensor) -> list:
        if len(disp_label.shape) == self.DISP_DIM_LEN:
            disp_label = disp_label.unsqueeze(self.ID_CHANNEL)

        _, _, h, w = disp_label.shape
        left_feat = F.interpolate(left_feat, [h, w], mode = 'bilinear', align_corners = False)
        right_feat = F.interpolate(right_feat, [h, w], mode = 'bilinear', align_corners = False)

        alignment_loss = self._alignment_loss(left_feat, right_feat, disp_label, mask_disp)
        matching_loss = self._feature_matching_loss(left_feat, right_feat, disp_label, mask_disp)

        return [alignment_loss + matching_loss, alignment_loss, matching_loss]
