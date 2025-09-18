# -*- coding: utf-8 -*-
"""3D matching head with full-resolution disparity output.

This module aggregates a grouped cost volume [B,C,D,H4,W4] with a stacked
hourglass (HG3D-Plus), then returns BOTH low-res disparity (H4,W4) and
original-resolution disparity (H,W). Two full-res modes are supported:

  - "disp-jbu": regress disparity at H4,W4 -> guided upsample (JBU) or bilinear.
  - "prob-dhw": upsample probability volume to (D*,H,W), renormalize over D,
                soft-argmin to get full-res disparity directly.
  - "prob-dhw-tiled": tile-wise variant of prob-dhw along width to reduce memory.

If your cost volume used a disparity band [dmin:dmax], pass band_offset
(= dmin * (W/W4)) so full-res disparity aligns to the global index/units.

All functions follow a Google-style and are pure PyTorch.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, List
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt

# Optional FeatUp JBU (https://github.com/mhamilton723/FeatUp)
try:
    from featup.upsamplers import JBULearnedRange
    _HAS_FEATUP = True
except Exception:
    _HAS_FEATUP = False


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def _soft_argmin(cost: torch.Tensor, temp: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute soft-argmin and probability from scalar cost.

    Args:
      cost: Scalar cost volume [B, D, H, W], lower=better.
      temp: Temperature for softmax over -cost.

    Returns:
      disp: [B, 1, H, W] expected disparity (indices 0..D-1).
      prob: [B, D, H, W] probability volume (sum_D = 1).
    """
    B, D, H, W = cost.shape
    prob = torch.softmax(-cost / max(temp, 1e-6), dim=1)
    d_vals = torch.arange(D, device=cost.device, dtype=cost.dtype).view(1, D, 1, 1)
    disp = (prob * d_vals).sum(1, keepdim=True)
    return disp, prob


def _interp3d_prob(
    P_lr: torch.Tensor,
    size_dhw: Tuple[int, int, int],
    *,
    amp: bool = True,
) -> torch.Tensor:
    """Trilinear upsample probability volume and re-normalize along D.

    Args:
      P_lr: [B, D_lr, H_lr, W_lr] probability (sum_D=1).
      size_dhw: Target (D_hr, H, W).
      amp: Use autocast for memory saving (inference).

    Returns:
      P_hr: [B, D_hr, H, W], re-normalized on dim=1.
    """
    B, D_lr, H_lr, W_lr = P_lr.shape
    D_hr, H, W = size_dhw
    x = P_lr.unsqueeze(1)  # [B,1,D_lr,H_lr,W_lr]
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (amp and torch.cuda.is_available()) else nullcontext()
    with ctx:
        P_hr = F.interpolate(x, size=(D_hr, H, W), mode="trilinear", align_corners=True).squeeze(1)
        P_hr = P_hr / (P_hr.sum(dim=1, keepdim=True) + 1e-6)
    return P_hr


def _fullres_from_prob_dhw(
    P_lr: torch.Tensor,
    orig_hw: Tuple[int, int],
    *,
    d_scale: Optional[float] = None,
    band_offset: float = 0.0,
    amp: bool = True,
) -> torch.Tensor:
    """Full-res disparity from P by upsampling in D/H/W and soft-argmin.

    Args:
      P_lr: [B, D_lr, H_lr, W_lr] probability at cost resolution.
      orig_hw: (H, W) original size.
      d_scale: Multiplier for D (if None, use W/W_lr).
      band_offset: Absolute disparity offset to add back (if banding).
      amp: autocast during interpolate to save memory.

    Returns:
      disp_full: [B, 1, H, W] in original pixel units.
    """
    B, D_lr, H_lr, W_lr = P_lr.shape
    H, W = orig_hw
    scale_x = float(W) / float(W_lr)
    if d_scale is None:
        d_scale = scale_x
    D_hr = max(1, int(round(D_lr * float(d_scale))))

    P_hr = _interp3d_prob(P_lr, size_dhw=(D_hr, H, W), amp=amp)
    d_vals = torch.arange(D_hr, device=P_hr.device, dtype=P_hr.dtype).view(1, D_hr, 1, 1)
    disp_full = (P_hr * d_vals).sum(1, keepdim=True) + float(band_offset)
    return disp_full


def _stitch_width(chunks: List[torch.Tensor], W_full: int, overlap: int) -> torch.Tensor:
    """Blend-stitch a list of tensors [B,C,D,H,W_i] along width."""
    assert len(chunks) > 0
    B, C, D, H, _ = chunks[0].shape
    out = chunks[0].new_zeros((B, C, D, H, W_full))
    weight = chunks[0].new_zeros((1, 1, 1, 1, W_full))
    cur = 0
    for i, t in enumerate(chunks):
        Wi = t.shape[-1]
        l = cur
        r = cur + Wi
        w = torch.ones((1, 1, 1, 1, Wi), device=t.device, dtype=t.dtype)
        if i > 0:
            ramp = torch.linspace(0, 1, steps=overlap, device=t.device, dtype=t.dtype)
            w[..., :overlap] = ramp
        if i < len(chunks) - 1:
            ramp = torch.linspace(1, 0, steps=overlap, device=t.device, dtype=t.dtype)
            w[..., -overlap:] = ramp
        out[..., l:r] += t * w
        weight[..., l:r] += w
        cur = r - overlap
    out = out / weight.clamp_min(1e-6)
    return out


def _fullres_from_prob_dhw_tiled(
    P_lr: torch.Tensor,
    orig_hw: Tuple[int, int],
    *,
    d_scale: Optional[float] = None,
    band_offset: float = 0.0,
    tile_w: int = 160,
    overlap: int = 24,
    amp: bool = True,
) -> torch.Tensor:
    """Tile-wise version of _fullres_from_prob_dhw to reduce memory.

    Args:
      P_lr: [B, D_lr, H_lr, W_lr]
      orig_hw: (H, W)
      d_scale: If None, use W/W_lr.
      band_offset: Add back absolute disparity offset (if banding).
      tile_w: Tile width on low-res W_lr (will be scaled internally).
      overlap: Linear blend size (on full-res width).
      amp: autocast interpolate.

    Returns:
      disp_full: [B,1,H,W]
    """
    B, D_lr, H_lr, W_lr = P_lr.shape
    H, W = orig_hw
    scale_x = float(W) / float(W_lr)
    if d_scale is None:
        d_scale = scale_x
    D_hr = max(1, int(round(D_lr * float(d_scale))))

    # Split along low-res width; upsample prob per tile to full-res, then stitch.
    tiles: List[torch.Tensor] = []
    cur = 0
    while cur < W_lr:
        r = min(cur + tile_w, W_lr)
        if r - cur < (tile_w // 3) and cur > 0:
            break
        P_tile = P_lr[..., cur:r]  # [B,D_lr,H_lr,W_tile]
        # Upsample to full-res width for this tile
        W_tile_full = int(round((r - cur) * scale_x))
        P_hr_tile = _interp3d_prob(P_tile, size_dhw=(D_hr, H, W_tile_full), amp=amp).unsqueeze(1)  # [B,1,D_hr,H,Wt_full]
        tiles.append(P_hr_tile)
        cur = r

    P_hr = _stitch_width(tiles, W_full=W, overlap=overlap).squeeze(1)  # [B,D_hr,H,W]
    P_hr = P_hr / (P_hr.sum(dim=1, keepdim=True) + 1e-6)
    d_vals = torch.arange(D_hr, device=P_hr.device, dtype=P_hr.dtype).view(1, D_hr, 1, 1)
    disp_full = (P_hr * d_vals).sum(1, keepdim=True) + float(band_offset)
    return disp_full


# ---------------------------------------------------------------------
# 3D blocks
# ---------------------------------------------------------------------
class Conv3dGN(nn.Module):
    """Conv3d + GroupNorm + ReLU with 'same' padding."""

    def __init__(self, in_ch: int, out_ch: int, k=(3, 3, 3), s=(1, 1, 1), groups_gn: int = 8) -> None:
        super().__init__()
        p = tuple(kk // 2 for kk in k)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.gn = nn.GroupNorm(num_groups=min(groups_gn, max(1, out_ch // 4)), num_channels=out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class Res3DBlock(nn.Module):
    """Residual 3D block with (1,3,3)+(3,3,3) kernels."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv1 = Conv3dGN(ch, ch, k=(1, 3, 3))
        self.conv2 = Conv3dGN(ch, ch, k=(3, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


class SE3D(nn.Module):
    """Squeeze-and-Excitation for [B,C,D,H,W]."""

    def __init__(self, ch: int, r: int = 8) -> None:
        super().__init__()
        hid = max(ch // r, 8)
        self.fc1 = nn.Linear(ch, hid, bias=False)
        self.fc2 = nn.Linear(hid, ch, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        s = x.mean(dim=(2, 3, 4))  # [B,C]
        w = torch.relu(self.fc1(s))
        w = torch.sigmoid(self.fc2(w)).view(B, C, 1, 1, 1)
        return x * w


class DispAttention(nn.Module):
    """Per-disparity attention (pool over H/W, 1x1 conv along D)."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.proj = nn.Conv1d(ch, ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        g = x.mean(dim=(3, 4))                 # [B,C,D]
        g = torch.sigmoid(self.proj(g)).view(B, C, D, 1, 1)
        return x * g


class Hourglass3D(nn.Module):
    """Anisotropic 3D hourglass for cost aggregation."""

    def __init__(self, in_ch: int, base: int = 40, depth: int = 2, use_ckpt: bool = True):
        super().__init__()
        self.use_ckpt = use_ckpt
        self.stem = nn.Sequential(Conv3dGN(in_ch, base, k=(3, 3, 3)), Res3DBlock(base))
        self.enc1_down = Conv3dGN(base, base * 2, k=(1, 3, 3), s=(1, 2, 2))   # H/2, W/2
        self.enc1_body = nn.Sequential(*[Res3DBlock(base * 2) for _ in range(depth)])
        self.enc2_down = Conv3dGN(base * 2, base * 4, k=(3, 3, 3), s=(2, 2, 2))  # D/2, H/4, W/4
        self.enc2_body = nn.Sequential(*[Res3DBlock(base * 4) for _ in range(depth)])
        self.bottleneck = nn.Sequential(Res3DBlock(base * 4), Res3DBlock(base * 4), SE3D(base * 4), DispAttention(base * 4))
        self.up2 = nn.Conv3d(base * 4, base * 2, kernel_size=1, bias=False)
        self.dec2_body = nn.Sequential(*[Res3DBlock(base * 2) for _ in range(depth)])
        self.up1 = nn.Conv3d(base * 2, base, kernel_size=1, bias=False)
        self.dec1_body = nn.Sequential(*[Res3DBlock(base) for _ in range(depth)])
        self.head = nn.Conv3d(base, 1, kernel_size=1, bias=False)

    @staticmethod
    def _upsample(x: torch.Tensor, size_dhw: Tuple[int, int, int]) -> torch.Tensor:
        D, H, W = size_dhw
        return F.interpolate(x, size=(D, H, W), mode="trilinear", align_corners=True)

    def _maybe_ckpt(self, fn, x):
        return ckpt(fn, x) if (self.use_ckpt and self.training) else fn(x)

    def forward(self, C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward a single hourglass.

        Args:
          C: [B, C_in, D, H, W].

        Returns:
          feat: [B, base, D, H, W].
          cost: [B, D, H, W].
        """
        x0 = self.stem(C)
        x1 = self._maybe_ckpt(self.enc1_down, x0)
        x1 = self._maybe_ckpt(self.enc1_body, x1)
        x2 = self._maybe_ckpt(self.enc2_down, x1)
        x2 = self._maybe_ckpt(self.enc2_body, x2)
        xb = self._maybe_ckpt(self.bottleneck, x2)
        y2 = self._upsample(self.up2(xb), (x1.shape[2], x1.shape[3], x1.shape[4])) + x1
        y2 = self._maybe_ckpt(self.dec2_body, y2)
        y1 = self._upsample(self.up1(y2), (x0.shape[2], x0.shape[3], x0.shape[4])) + x0
        y1 = self._maybe_ckpt(self.dec1_body, y1)
        cost = self.head(y1).squeeze(1)
        return y1, cost


# ---------------------------------------------------------------------
# Full-res guided upsampler (JBU x2 -> H,W), fallback to bilinear
# ---------------------------------------------------------------------
class GuidedDispUpsampler(nn.Module):
    """Edge-aware disparity upsampler from (H4,W4) to (H,W).

    If FeatUp is available and guidance is given, apply two 2× JBU steps.
    Otherwise fall back to bilinear interpolation. Always rescale disparity
    units along width by (W/W4).
    """

    def __init__(self, use_featup: bool = True, jbu_dim: int = 16, radius: int = 3) -> None:
        super().__init__()
        self.use_featup = use_featup and _HAS_FEATUP
        if self.use_featup:
            self.jbu1 = JBULearnedRange(guidance_dim=3, feat_dim=1, key_dim=jbu_dim, radius=radius)
            self.jbu2 = JBULearnedRange(guidance_dim=3, feat_dim=1, key_dim=jbu_dim, radius=radius)

    @staticmethod
    def _resize_guidance(g: torch.Tensor, h: int, w: int) -> torch.Tensor:
        return F.interpolate(g, size=(h, w), mode="bilinear", align_corners=True)

    def forward(self, disp_lr: torch.Tensor, guidance: Optional[torch.Tensor], orig_hw: Tuple[int, int]) -> torch.Tensor:
        """Upsample to original size and rescale disparity units.

        Args:
          disp_lr:  [B,1,H4,W4] disparity at low-res pixels.
          guidance: [B,3,H,W] guidance (RGB). If None or FeatUp missing -> bilinear.
          orig_hw:  (H,W) original size.

        Returns:
          disp_full: [B,1,H,W] disparity in original pixel units.
        """
        B, _, Hlr, Wlr = disp_lr.shape
        H, W = orig_hw
        scale_x = float(W) / float(Wlr)

        if self.use_featup and (guidance is not None):
            g2 = self._resize_guidance(guidance, Hlr * 2, Wlr * 2)
            x2 = self.jbu1(disp_lr, g2)
            g4 = self._resize_guidance(guidance, H, W)
            x4 = self.jbu2(x2, g4)
            disp_full = x4 * scale_x
        else:
            disp_full = F.interpolate(disp_lr, size=(H, W), mode="bilinear", align_corners=True) * scale_x
        return disp_full


# ---------------------------------------------------------------------
# HG3D-Plus with full-res outputs
# ---------------------------------------------------------------------
class HG3DPlus(nn.Module):
    """Stacked hourglass 3D aggregator with original-size disparity output.

    Pipeline:
      HG1(C) -> cost1 -> inj -> C2 -> HG2(C2) -> cost
      -> disp_lr/prob at H4,W4
      -> full-res 'disp_full' via one of:
         - "disp-jbu": guided upsampler (JBU or bilinear)
         - "prob-dhw": 3D probability upsample to (D*,H,W), soft-argmin
         - "prob-dhw-tiled": tile-wise 3D prob upsample to reduce memory

    Args:
      in_ch:        Input channels C of cost volume.
      base, depth:  Hourglass width/depth.
      use_ckpt:     Enable checkpointing (train only).
      temperature1/2: Softmax temperature for stage-1/final.
      residual_gain: Scale for residual injection (cost1 -> C).
      fuse_average: If True, fuse (cost1+cost2)/2; else use cost2.
      make_fullres: Create internal upsampler when outputting full-res.
      use_featup:   JBU availability for "disp-jbu".
      fullres_mode: "disp-jbu" (default) | "prob-dhw" | "prob-dhw-tiled".
      tile_w/overlap: params for "prob-dhw-tiled".
    """

    def __init__(self,
                 in_ch: int = 64,
                 base: int = 40,
                 depth: int = 2,
                 use_ckpt: bool = True,
                 temperature1: float = 1.2,
                 temperature2: float = 0.9,
                 residual_gain: float = 0.5,
                 fuse_average: bool = True,
                 make_fullres: bool = True,
                 use_featup: bool = True,
                 fullres_mode: str = "disp-jbu",
                 tile_w: int = 160,
                 overlap: int = 24) -> None:
        super().__init__()
        self.temperature1 = temperature1
        self.temperature2 = temperature2
        self.residual_gain = residual_gain
        self.fuse_average = fuse_average
        self.fullres_mode = fullres_mode
        self.tile_w = tile_w
        self.overlap = overlap

        self.hg1 = Hourglass3D(in_ch=in_ch, base=base, depth=depth, use_ckpt=use_ckpt)
        self.inject = nn.Conv3d(1, in_ch, kernel_size=1, bias=False)
        self.hg2 = Hourglass3D(in_ch=in_ch, base=base, depth=depth, use_ckpt=use_ckpt)

        self.fullres_up = GuidedDispUpsampler(use_featup=use_featup) if make_fullres else None

    def forward(self,
                C: torch.Tensor,
                *,
                orig_hw: Optional[Tuple[int, int]] = None,
                guidance: Optional[torch.Tensor] = None,
                band_offset: float = 0.0) -> Dict[str, torch.Tensor]:
        """Aggregate cost and (optionally) output original-size disparity.

        Args:
          C:          [B, C, D, H4, W4] cost (lower=better).
          orig_hw:    (H,W) original size. If provided, return 'disp_full'.
          guidance:   [B,3,H,W] guidance image for "disp-jbu".
          band_offset: Absolute disparity offset to add back in full-res
                       (e.g., dmin * (W/W4) if banded cost was used).

        Returns:
          dict with:
            'disp_lr':   [B,1,H4,W4] disparity at cost resolution.
            'disp_full': [B,1,H,W]   (if orig_hw given)
            'prob':      [B,D,H4,W4]
            'cost':      [B,D,H4,W4]
            'aux':       stage-1 outputs
        """
        # Stage-1 hourglass
        feat1, cost1 = self.hg1(C)
        disp1, prob1 = _soft_argmin(cost1, temp=self.temperature1)

        # Residual injection
        inj = self.inject(cost1.unsqueeze(1))  # [B,C,D,H4,W4]
        C2 = C + self.residual_gain * inj

        # Stage-2 hourglass
        _, cost2 = self.hg2(C2)
        cost = 0.5 * (cost1 + cost2) if self.fuse_average else cost2

        disp_lr, prob = _soft_argmin(cost, temp=self.temperature2)

        out = {
            "disp_lr": disp_lr,
            "prob": prob,
            "cost": cost,
            "aux": {"disp1": disp1, "prob1": prob1, "cost1": cost1},
        }

        # Full-resolution branch
        if orig_hw is not None:
            H4, W4 = disp_lr.shape[-2:]
            if self.fullres_mode == "prob-dhw":
                out["disp_full"] = _fullres_from_prob_dhw(
                    prob, orig_hw, d_scale=None, band_offset=band_offset, amp=not self.training
                )
            elif self.fullres_mode == "prob-dhw-tiled":
                out["disp_full"] = _fullres_from_prob_dhw_tiled(
                    prob, orig_hw, d_scale=None, band_offset=band_offset,
                    tile_w=self.tile_w, overlap=self.overlap, amp=not self.training
                )
            else:  # "disp-jbu"
                if self.fullres_up is not None:
                    out["disp_full"] = self.fullres_up(disp_lr, guidance, orig_hw) + float(band_offset)
                else:
                    H, W = orig_hw
                    scale_x = float(W) / float(W4)
                    out["disp_full"] = F.interpolate(disp_lr, size=(H, W), mode="bilinear", align_corners=True) * scale_x + float(band_offset)

        return out
