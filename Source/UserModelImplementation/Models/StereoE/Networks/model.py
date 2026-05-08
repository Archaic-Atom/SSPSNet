# -*- coding: utf-8 -*-
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from UserModelImplementation.Models.StereoA.Networks import build_gwc_volume
from UserModelImplementation.Models.StereoC.Networks._sparse_prompt_head import SparsePromptFull

from ._backbone import build_backbone
from ._lite_matcher import LiteMatchingHead


class StereoE(nn.Module):
    """Fast sparse-prompt stereo variant."""

    NUM_SCALES = 4
    GROUPS = 32
    PROJ_CHANNEL = 256
    PROMPT_DOWNSAMPLE = 4

    def __init__(self, in_channels: int, start_disp: int, disp_num: int,
                 backbone: str = 'dinov3', backbone_variant: str = None,
                 backbone_weights: str = None, pre_train_opt: bool = False,
                 confidence_level: float = 0.65, prompt_min_conf: float = 0.65) -> None:
        super().__init__()
        self.start_disp = start_disp
        self.disp_num = disp_num
        self.pre_train_opt = pre_train_opt
        self.confidence_level = confidence_level
        self.prompt_min_conf = prompt_min_conf
        self._h, self._w = None, None

        self.backbone = build_backbone(backbone, backbone_variant, backbone_weights, freeze=True)
        self.projects = nn.ModuleList([
            nn.Sequential(
                nn.LazyConv2d(self.PROJ_CHANNEL, kernel_size=1, bias=False),
                nn.GroupNorm(16, self.PROJ_CHANNEL),
                nn.GELU(),
            ) for _ in range(self.NUM_SCALES)
        ])
        fusion_in = self.PROJ_CHANNEL * self.NUM_SCALES
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            nn.Conv2d(256, 64, kernel_size=1, bias=False),
            nn.GELU(),
        )

        self.matcher = LiteMatchingHead(self.GROUPS, start_disp, disp_num, base_channels=64, num_blocks=2)
        self.prompt_head = SparsePromptFull(mode="prob-dhw", anchor_mode="prob-dhw", use_featup=True)

    def _feature_extraction_module_proc(self, left_img: torch.Tensor,
                                        right_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        left_feats = self._slice_or_pad(self.backbone(left_img))
        right_feats = self._slice_or_pad(self.backbone(right_img))
        target_hw = left_feats[-1].shape[-2:]

        proc_left, proc_right = [], []
        for proj, l_feat, r_feat in zip(self.projects, left_feats, right_feats):
            l_proj = proj(l_feat)
            r_proj = proj(r_feat)
            if l_proj.shape[-2:] != target_hw:
                l_proj = F.interpolate(l_proj, size=target_hw, mode='bilinear', align_corners=False)
                r_proj = F.interpolate(r_proj, size=target_hw, mode='bilinear', align_corners=False)
            proc_left.append(l_proj)
            proc_right.append(r_proj)

        left_feat = self.fusion(torch.cat(proc_left, dim=1))
        right_feat = self.fusion(torch.cat(proc_right, dim=1))
        return left_feat, right_feat

    def _slice_or_pad(self, feats):
        feats = list(feats)
        if len(feats) >= self.NUM_SCALES:
            return feats[:self.NUM_SCALES]
        last = feats[-1]
        for _ in range(self.NUM_SCALES - len(feats)):
            feats.append(last)
        return feats

    def _build_cost_volume_proc(self, left_feat: torch.Tensor,
                                right_feat: torch.Tensor) -> torch.Tensor:
        return build_gwc_volume(left_feat, right_feat, self.start_disp,
                                self.disp_num, self.GROUPS)

    def _match_and_prompt(self, cost: torch.Tensor,
                          left_img: torch.Tensor) -> tuple:
        match = self.matcher(cost)
        prob = match["prob"]
        logits = match["logits"]
        aux_logits = match["aux_logits"]
        disp = match["disp"]
        disp_aux = match["aux_disp"]
        confidence = match["confidence"]

        prob_full = F.interpolate(prob, size=(self._h, self._w), mode='bilinear', align_corners=False)
        prob_full = prob_full / (prob_full.sum(dim=1, keepdim=True) + 1e-6)

        H4 = max(1, self._h // self.PROMPT_DOWNSAMPLE)
        W4 = max(1, self._w // self.PROMPT_DOWNSAMPLE)
        prob_lr = F.interpolate(prob, size=(H4, W4), mode='bilinear', align_corners=False)
        prob_lr = prob_lr / (prob_lr.sum(dim=1, keepdim=True) + 1e-6)

        prompt_out = self.prompt_head(
            prob_lr, orig_hw=(self._h, self._w),
            guidance=left_img, min_conf=self.prompt_min_conf,
            band_offset=float(self.start_disp)
        )
        disp_full = prompt_out["disp_full"]

        disp_coarse = F.interpolate(disp, size=(self._h, self._w), mode='bilinear', align_corners=True)
        disp_aux = F.interpolate(disp_aux, size=(self._h, self._w), mode='bilinear', align_corners=True)

        return {
            "disp_full": disp_full,
            "disp_init": prompt_out["disp_init"],
            "disp_coarse": disp_coarse,
            "disp_aux": disp_aux,
            "prob_prompt": prob_lr,
            "prob_full": prob_full,
            "prob_logits": F.interpolate(logits, size=(self._h, self._w), mode='bilinear', align_corners=False),
            "prob_aux_logits": F.interpolate(aux_logits, size=(self._h, self._w), mode='bilinear', align_corners=False),
            "confidence": F.interpolate(confidence, size=(self._h, self._w), mode='bilinear', align_corners=False),
            "anchors": prompt_out["anchors"],
            "uspf": prompt_out["uspf"],
            "band_offset_px": float(self.start_disp),
        }, disp_full

    def _mask_pre_train_proc(self, left_img: torch.Tensor,
                             right_img: torch.Tensor) -> torch.Tensor:
        return list(self._feature_extraction_module_proc(left_img, right_img))

    def forward(self, left_img: torch.Tensor, right_img: torch.Tensor) -> list:
        if self.pre_train_opt:
            return self._mask_pre_train_proc(left_img, right_img)

        self._h, self._w = left_img.shape[-2], left_img.shape[-1]
        left_feat, right_feat = self._feature_extraction_module_proc(left_img, right_img)
        cost = self._build_cost_volume_proc(left_feat, right_feat)
        match_out, disp_full = self._match_and_prompt(cost, left_img)
        return [match_out, disp_full]
