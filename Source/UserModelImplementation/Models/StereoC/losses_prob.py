# -*- coding: utf-8 -*-
"""Probability-volume supervision at 1/4 resolution with band support.

Supervises stage-1 `prob1` and final `prob` at H/4 × W/4:
  - soft windowed cross-entropy around GT bin (±radius, triangular)
  - optional entropy regularization (sharpen peaks)
  - precise GT→bin mapping under disparity banding (dmin:dmax)

You may pass either:
  - band_dmin: low-res bin start index (int, in [0, ...])
  - or band_offset_pixels: full-res pixel offset used in forward (= dmin * (W/W4))

Exactly one of them should be provided (band_dmin preferred).
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_dim(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        B, C, H, W = x.shape
    elif x.dim() == 3:
        # [B, H, W] -> [B,1,H,W]
        B, H, W = x.shape
        x = x.unsqueeze(1)
    elif x.dim() == 2:
        # [H, W] -> [1,1,H,W]
        H, W = x.shape
        x = x.unsqueeze(0).unsqueeze(0)
        B, C = 1, 1
    else:
        raise ValueError(f"Expected 2D/3D/4D disparity, got shape {tuple(x.shape)}")

    # 明确拒绝展平的形状（例如 [B,1,HW] 或 [B,1,H]）
    if x.shape[-1] == 1 or x.shape[-2] == 1:
        raise ValueError(
            f"Disparity looks flattened or missing a spatial dim: {tuple(x.shape)}. "
            f"Make sure it is [B,1,H,W] (not [B,1,HW] or [B,1,H])."
        )
    return x


def _downsample_disp_to_quarter(d_full: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """Area downsample disparity to H/4,W/4 (keeps values in FULL-RES pixels)."""
    H4, W4 = target_hw
    d_full = _check_dim(d_full)
    return F.interpolate(d_full, size=(H4, W4), mode="area")


def _compute_band_dmin(
    *,
    band_dmin: Optional[int],
    band_offset_pixels: Optional[float],
    W_full: int,
    W4: int,
) -> float:
    """Return dmin in LOW-RES pixel units.

    If band_dmin is given, return it as float.
    Else infer from band_offset_pixels used in forward:
      band_offset_pixels = dmin * (W_full / W4)  =>  dmin = band_offset_pixels * (W4 / W_full).
    """
    if band_dmin is not None:
        return float(band_dmin)
    if band_offset_pixels is not None:
        return float(band_offset_pixels) * (float(W4) / float(W_full))
    # no band: start at 0
    return 0.0


def _gt_to_bins_with_band(
    d_q_full_units: torch.Tensor,   # [B,1,H4,W4] (FULL-RES pixels)
    *,
    D: int,
    W_full: int,
    W4: int,
    band_dmin: Optional[int],
    band_offset_pixels: Optional[float],
) -> torch.Tensor:
    """Map H/4 GT disparity (full-res units) to fractional BIN indices under banding.

    y_bin = (d_full / (W_full/W4)) - dmin
          = d_lowres - dmin,  where d_lowres in LOW-RES pixel units.
    """
    scale_lr = float(W4) / float(W_full)          # full-res px -> low-res px
    d_lr = d_q_full_units * scale_lr              # [B,1,H4,W4], low-res units
    dmin = _compute_band_dmin(band_dmin=band_dmin, band_offset_pixels=band_offset_pixels,
                              W_full=W_full, W4=W4)
    y_bin = d_lr - dmin                           # fractional bin
    # 不要硬裁剪；让三角窗处理边缘，但也产出 in-band mask 供加权。
    return y_bin


def _inband_weight(
    y_bin: torch.Tensor,  # [B,1,H4,W4]
    D: int,
    radius: int,
    soften: float = 2.0,
) -> torch.Tensor:
    """Compute a [0,1] weight telling how much the GT falls inside the [0,D-1] band.

    We softly downweight pixels outside the band (by distance to [0, D-1]).
    soften: slope for a logistic 'soft clamp' (bigger => sharper).
    """
    B, _, H4, W4 = y_bin.shape
    # distance to interval [0, D-1]
    left_over = (-y_bin).clamp_min(0.0)              # y_bin < 0
    right_over = (y_bin - (D - 1)).clamp_min(0.0)     # y_bin > D-1
    dist = (left_over + right_over)                   # how far outside
    # allow radius margin: if within ±radius of the band edges, keep weight ~1
    dist = (dist - float(radius)).clamp_min(0.0)
    # soft weighting: exp(-soften * dist)
    w = torch.exp(-soften * dist)
    return w  # [B,1,H4,W4], 1 inside band, decays smoothly outside


def _triangular_soft_labels(
    y_bin: torch.Tensor,  # [B,1,H4,W4]
    D: int,
    radius: int = 2,
) -> torch.Tensor:
    """Triangular labels centered at y_bin within ±radius, renormalized over D."""
    B, _, H4, W4 = y_bin.shape
    device, dtype = y_bin.device, y_bin.dtype
    d_vals = torch.arange(D, device=device, dtype=dtype).view(1, D, 1, 1)
    dist = (d_vals - y_bin).abs()
    T = (radius + 1 - dist).clamp(min=0.0)
    T = T / (T.sum(1, keepdim=True) + 1e-6)
    return T  # [B,D,H4,W4]


def _soft_ce(P: torch.Tensor, T: torch.Tensor, weight: Optional[torch.Tensor]) -> torch.Tensor:
    """Soft-label cross-entropy: - sum_d T * log P, weighted by 'weight' (B,1,H4,W4)."""
    logP = (P.clamp_min(1e-8)).log()
    ce_map = -(T * logP).sum(1, keepdim=True)   # [B,1,H4,W4]
    if weight is None:
        return ce_map.mean()
    num = (ce_map * weight).sum()
    den = weight.sum().clamp_min(1e-6)
    return num / den


def _entropy(P: torch.Tensor, weight: Optional[torch.Tensor]) -> torch.Tensor:
    """Entropy of P (minimize to sharpen)."""
    ent_map = (P * P.clamp_min(1e-8).log()).sum(1, keepdim=True)  # [B,1,H4,W4]
    if weight is None:
        return ent_map.mean()
    return (ent_map * weight).sum() / weight.sum().clamp_min(1e-6)


class ProbVolumeLoss(nn.Module):
    """Supervise H/4 probability volumes (`prob1`, `prob`) with band support.

    Args:
      radius:    triangular window half-size (±radius bins).
      w_prob:    weight for final prob CE.
      w_prob1:   weight for stage-1 prob1 CE.
      w_entropy: weight for entropy regularization on final prob.
      soften_oob: softness for out-of-band downweighting (exp(-k*dist)).

    Usage:
      Lp = ProbVolumeLoss(...)(prob=ret["prob"],
                               prob1=ret["aux"]["prob1"],
                               disp_gt_full=gt, mask_full=(gt>0),
                               orig_hw=(H,W), H4W4=(H4,W4), D=prob.shape[1],
                               band_dmin=dmin,          # prefer passing this
                               band_offset_pixels=None) # or pass this instead
    """

    def __init__(self,
                 radius: int = 2,
                 w_prob: float = 0.02,
                 w_prob1: float = 0.02,
                 w_entropy: float = 0.01,
                 soften_oob: float = 2.0) -> None:
        super().__init__()
        self.radius = int(radius)
        self.w_prob = float(w_prob)
        self.w_prob1 = float(w_prob1)
        self.w_entropy = float(w_entropy)
        self.soften_oob = float(soften_oob)

    def forward(self,
                *,
                prob: torch.Tensor,                    # [B,D,H4,W4]
                prob1: Optional[torch.Tensor],         # [B,D,H4,W4] or None
                disp_gt_full: torch.Tensor,            # [B,1,H,W] (FULL-RES px)
                mask_full: torch.Tensor,               # [B,1,H,W] (1=valid)
                orig_hw: Tuple[int, int],              # (H, W)
                H4W4: Tuple[int, int],                 # (H4, W4)
                D: int,
                band_dmin: Optional[int] = None,
                band_offset_pixels: Optional[float] = None
                ) -> Dict[str, torch.Tensor]:
        H, W = orig_hw
        H4, W4 = H4W4

        # --- 原先步骤 1~2 保持不变 ---
        gt_q = _downsample_disp_to_quarter(disp_gt_full, (H4, W4))                     # [B,1,H4,W4]
        mask_full = _check_dim(mask_full)
        mk_q = F.interpolate(mask_full.float(), size=(H4, W4), mode="nearest")         # [B,1,H4,W4]
        y_bin = _gt_to_bins_with_band(
            gt_q, D=D, W_full=W, W4=W4,
            band_dmin=band_dmin, band_offset_pixels=band_offset_pixels
        )  # [B,1,H4,W4]

        # ------ 新增：分别用各自的 D 计算 Tp/wp 和 T1/w1 ------
        # final prob 分支
        D_p = prob.shape[1]
        T_p = _triangular_soft_labels(y_bin, D=D_p, radius=self.radius)                           # [B,Dp,H4,W4]
        w_p = _inband_weight(y_bin, D=D_p, radius=self.radius, soften=self.soften_oob) * mk_q     # [B,1,H4,W4]
        L_prob = _soft_ce(prob, T_p, weight=w_p)

        # stage-1 prob1 分支（可选）
        if (prob1 is not None) and (self.w_prob1 > 0.0):
            D_1 = prob1.shape[1]
            T_1 = _triangular_soft_labels(y_bin, D=D_1, radius=self.radius)                        # [B,D1,H4,W4]
            w_1 = _inband_weight(y_bin, D=D_1, radius=self.radius, soften=self.soften_oob) * mk_q  # [B,1,H4,W4]
            L_prob1 = _soft_ce(prob1, T_1, weight=w_1)
        else:
            L_prob1 = prob.new_tensor(0.0)

        # 熵正则用 final prob 的 D
        L_ent = _entropy(prob, weight=w_p) if self.w_entropy > 0.0 else prob.new_tensor(0.0)

        loss = self.w_prob * L_prob + self.w_prob1 * L_prob1 + self.w_entropy * L_ent
        return {
            "loss": loss,
            "L_prob": L_prob.detach(),
            "L_prob1": L_prob1.detach(),
            "L_ent": L_ent.detach(),
            "w_inband_mean": w_p.mean().detach(),
        }
