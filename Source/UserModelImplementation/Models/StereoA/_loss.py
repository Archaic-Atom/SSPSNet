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

    def matching_loss(self, disp_list: list, disp_label: torch.Tensor,
                      mask_disp: torch.Tensor, udc: bool) -> torch.Tensor:
        args = self.__arg
        res = []
        gt_distribute = self._disp2distribute(args.start_disp, disp_label, args.disp_num, b=2)

        loss_1 = 0.5 * F.smooth_l1_loss(disp_list[0][mask_disp], disp_label[mask_disp]) + \
            0.7 * F.smooth_l1_loss(disp_list[1][mask_disp], disp_label[mask_disp]) + \
            F.smooth_l1_loss(disp_list[2][mask_disp], disp_label[mask_disp])

        if udc:
            loss_2 = 0.5 * self._celoss(
                args.start_disp, disp_label, args.disp_num, gt_distribute, disp_list[3]) + \
                0.7 * self._celoss(
                    args.start_disp, disp_label, args.disp_num, gt_distribute, disp_list[4]) + \
                self._celoss(
                    args.start_disp, disp_label, args.disp_num, gt_distribute, disp_list[5])
            res.append(loss_1 + loss_2)
            res.append(loss_2)
        res.append(loss_1)

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
