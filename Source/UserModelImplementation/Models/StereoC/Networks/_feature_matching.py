# -*- coding: utf-8 -*-
"""HG3DPlus with band selection and full-resolution disparity outputs.

Features:
  - Two-stage 3D hourglass with residual cost injection.
  - Stage-1 prob -> GLOBAL band [d0,d1] -> Stage-2 only aggregates within band.
  - Soft-argmin on the sub-band; adds d0 offset automatically.
  - Three full-res modes:
      * "disp-jbu": regress at H/4,W/4 then (JBU/bilinear) upsample to H,W.
      * "prob-dhw": upsample prob in D/H/W to full-res; renormalize & soft-argmin.
      * "prob-dhw-tiled": width-tiled variant of prob-dhw to reduce memory.
  - Band offset is consistently propagated to full-res ("prob-*") and returned.

All outputs use ORIGINAL pixel units at full-res. Low-res disparity uses index units.
Google-style docstrings; no extraneous dependencies (FeatUp JBU is optional).
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
# Utilities
# ---------------------------------------------------------------------
def _soft_argmin(cost: torch.Tensor, temp: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute soft-argmin and probability from scalar cost.

    Args:
      cost: [B, D, H, W], lower=better.
      temp: Temperature for softmax over -cost.

    Returns:
      disp: [B, 1, H, W] expected disparity in index units.
      prob: [B, D, H, W] probability (sum over D = 1).
    """
    B, D, H, W = cost.shape
    prob = torch.softmax(-cost / max(temp, 1e-6), dim=1)
    d_vals = torch.arange(D, device=cost.device, dtype=cost.dtype).view(1, D, 1, 1)
    disp = (prob * d_vals).sum(1, keepdim=True)
    return disp, prob


def _conf_map_from_prob(P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """From prob [B,D,H,W], build confidence and argmax disparity.

    Confidence = pmax * tanh(pmax/p2), where p2 is 2nd peak (or 0).
    """
    pmax, d_hat = P.max(dim=1)  # [B,H,W], [B,H,W]
    top2 = torch.topk(P, k=min(2, P.shape[1]), dim=1).values
    p2 = top2[:, 1] if P.shape[1] >= 2 else torch.zeros_like(pmax)
    psr = pmax / (p2 + 1e-6)
    conf = (pmax * psr.tanh()).clamp(0, 1)
    return conf, d_hat


def _band_from_prob(
    prob: torch.Tensor,
    *,
    min_conf: float = 0.65,
    pad: int = 4,
    min_width: int = 16,
    min_points: int = 256,
) -> Tuple[int, int]:
    """Estimate a GLOBAL disparity band [d0,d1] from stage-1 prob.

    Strategy:
      - Build conf & argmax maps.
      - Per-image select pixels with conf>=min_conf; collect min/max d_hat.
      - Merge across batch -> [d0,d1], expand by pad, enforce min_width.
      - If too few points, fallback to full band.

    Args:
      prob: [B,D,H,W] stage-1 probability.
    """
    B, D, H, W = prob.shape
    conf, d_hat = _conf_map_from_prob(prob)
    dmins, dmaxs, n_keep = [], [], 0
    for b in range(B):
        mask = (conf[b] >= min_conf)
        n_keep += int(mask.sum().item())
        if mask.any():
            dmins.append(int(d_hat[b][mask].amin().item()))
            dmaxs.append(int(d_hat[b][mask].amax().item()))
        else:
            dmins.append(0)
            dmaxs.append(D - 1)

    # Fallback to full band if anchors are too few.
    if n_keep < min_points:
        return 0, D - 1

    d0 = max(0, min(dmins) - pad)
    d1 = min(D - 1, max(dmaxs) + pad)
    if d1 - d0 + 1 < min_width:
        c = (d0 + d1) // 2
        half = (min_width - 1) // 2
        d0 = max(0, c - half)
        d1 = min(D - 1, d0 + min_width - 1)
    return int(d0), int(d1)


def _interp3d_prob(
    P_lr: torch.Tensor,
    size_dhw: Tuple[int, int, int],
    *,
    amp: bool = True,
) -> torch.Tensor:
    """Trilinear upsample probability volume to (D,H,W) and renormalize dim=1."""
    x = P_lr.unsqueeze(1)  # [B,1,Dlr,Hlr,Wlr]
    use_amp = amp and torch.cuda.is_available()
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    with ctx:
        P_hr = F.interpolate(x, size=size_dhw, mode="trilinear", align_corners=True).squeeze(1)
        P_hr = P_hr / (P_hr.sum(dim=1, keepdim=True) + 1e-6)
    return P_hr


def _stitch_width(chunks: List[torch.Tensor], W_full: int, overlap: int) -> torch.Tensor:
    """Blend-stitch [B,C,D,H,W_i] tiles along width with linear ramps."""
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
        if i > 0 and overlap > 0:
            ramp = torch.linspace(0, 1, steps=overlap, device=t.device, dtype=t.dtype)
            w[..., :overlap] = ramp
        if i < len(chunks) - 1 and overlap > 0:
            ramp = torch.linspace(1, 0, steps=overlap, device=t.device, dtype=t.dtype)
            w[..., -overlap:] = ramp
        out[..., l:r] += t * w
        weight[..., l:r] += w
        cur = r - overlap
    out = out / weight.clamp_min(1e-6)
    return out


def _fullres_from_prob_dhw(
    P_lr: torch.Tensor,
    orig_hw: Tuple[int, int],
    *,
    d_scale: Optional[float] = None,
    band_offset: float = 0.0,
    amp: bool = True,
) -> torch.Tensor:
    """Full-res disparity from prob by upsampling D/H/W then soft-argmin.

    Args:
      P_lr:       [B,D_lr,H_lr,W_lr] (sub-band prob if band enabled).
      orig_hw:    (H,W) original image size.
      d_scale:    Disparity-axis scale; if None, use W/W_lr.
      band_offset:Absolute disparity offset to add back (in full-res pixels).
      amp:        autocast interpolate for memory saving (inference recommended).
    """
    B, D_lr, H_lr, W_lr = P_lr.shape
    H, W = orig_hw
    scale_x = float(W) / float(W_lr)
    if d_scale is None:
        d_scale = scale_x
    D_hr = max(1, int(round(D_lr * float(d_scale))))

    P_hr = _interp3d_prob(P_lr, size_dhw=(D_hr, H, W), amp=amp)        # [B,D_hr,H,W]
    d_vals = torch.arange(D_hr, device=P_hr.device, dtype=P_hr.dtype).view(1, D_hr, 1, 1)
    disp_full = (P_hr * d_vals).sum(1, keepdim=True) + float(band_offset)
    return disp_full


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
    """Width-tiled version of _fullres_from_prob_dhw to reduce memory."""
    B, D_lr, H_lr, W_lr = P_lr.shape
    H, W = orig_hw
    scale_x = float(W) / float(W_lr)
    if d_scale is None:
        d_scale = scale_x
    D_hr = max(1, int(round(D_lr * float(d_scale))))

    tiles: List[torch.Tensor] = []
    cur = 0
    while cur < W_lr:
        r = min(cur + tile_w, W_lr)
        if r - cur < (tile_w // 3) and cur > 0:
            break
        P_tile = P_lr[..., cur:r]  # [B,D_lr,H_lr,Wt]
        Wt_full = int(round((r - cur) * scale_x))
        P_hr_tile = _interp3d_prob(P_tile, size_dhw=(D_hr, H, Wt_full), amp=amp).unsqueeze(1)  # [B,1,D_hr,H,Wt_full]
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
    """Per-disparity attention: pool H/W then 1×1 conv over D."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.proj = nn.Conv1d(ch, ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        g = x.mean(dim=(3, 4))  # [B,C,D]
        g = torch.sigmoid(self.proj(g)).view(B, C, D, 1, 1)
        return x * g


class Hourglass3D(nn.Module):
    """Anisotropic 3D hourglass for cost aggregation."""

    def __init__(self, in_ch: int, base: int = 40, depth: int = 2, use_ckpt: bool = True):
        super().__init__()
        self.use_ckpt = use_ckpt
        self.stem = nn.Sequential(Conv3dGN(in_ch, base, k=(3, 3, 3)), Res3DBlock(base))
        self.enc1_down = Conv3dGN(base, base * 2, k=(1, 3, 3), s=(1, 2, 2))    # H/2, W/2
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
        """Args:
            C: [B, C_in, D, H, W]
           Returns:
            feat: [B, base, D, H, W]
            cost: [B, D, H, W]
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
        cost = self.head(y1).squeeze(1)  # [B,D,H,W]
        return y1, cost


# ---------------------------------------------------------------------
# Full-res guided upsampler (JBU x2 -> H,W), fallback to bilinear
# ---------------------------------------------------------------------
class GuidedDispUpsampler(nn.Module):
    """Edge-aware disparity upsampler from (H/4,W/4) to (H,W).

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
        # Guidance resize不需要 align_corners=True
        return F.interpolate(g, size=(h, w), mode="bilinear", align_corners=False)

    def forward(self, disp_lr: torch.Tensor, guidance: Optional[torch.Tensor], orig_hw: Tuple[int, int]) -> torch.Tensor:
        """Upsample and rescale disparity to original pixel units."""
        B, _, Hlr, Wlr = disp_lr.shape
        H, W = orig_hw
        scale_x = float(W) / float(Wlr)

        if self.use_featup and (guidance is not None):
            g2 = self._resize_guidance(guidance, Hlr * 2, Wlr * 2)
            x2 = self.jbu1(disp_lr, g2)
            g4 = self._resize_guidance(guidance, H, W)
            x4 = self.jbu2(x2, g4)
            return x4 * scale_x

        return F.interpolate(disp_lr, size=(H, W), mode="bilinear", align_corners=True) * scale_x


# ---------------------------------------------------------------------
# HG3D-Plus with band & full-res outputs
# ---------------------------------------------------------------------
class HG3DPlus(nn.Module):
    """Stacked hourglass 3D aggregator with band selection and full-res output.

    Pipeline:
      HG1(C) -> cost1 -> prob1 -> GLOBAL band [d0,d1]
      -> C_sub = C[:,:,d0:d1] ; inj_sub = Conv3d(cost1)[:,:,d0:d1]
      -> HG2(C_sub + w*inj_sub) -> cost_sub
      -> soft-argmin on subrange -> disp_lr_local + d0 (index units)
      -> full-res 'disp_full':
          - "disp-jbu": guided upsampler (JBU or bilinear) * (W/W4)
          - "prob-dhw": 3D prob upsample to (D*,H,W) + band_offset
          - "prob-dhw-tiled": tiled 3D upsample + band_offset

    Args:
      in_ch:        Input channels C of cost volume.
      base, depth:  Hourglass width/depth.
      use_ckpt:     Enable checkpointing (train only).
      temperature1/2: Softmax temperature for stage-1/final.
      residual_gain: Scale for residual injection (cost1 -> C).
      fuse_average: 0.5*(cost1_sub+cost2_sub) if True else cost2_sub.
      make_fullres: Create internal upsampler when outputting full-res.
      use_featup:   Whether to use JBU for "disp-jbu".
      fullres_mode: "disp-jbu" (default) | "prob-dhw" | "prob-dhw-tiled".
      tile_w/overlap: params for "prob-dhw-tiled".
      use_band:     Enable band selection between HG1/HG2.
      band_pad:     Pad (bins) around [min,max] of confident d_hat.
      band_min_width: Minimum sub-band width (bins).
      band_conf:    Confidence threshold to collect d_hat.
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
                 overlap: int = 24,
                 use_band: bool = True,
                 band_pad: int = 4,
                 band_min_width: int = 16,
                 band_conf: float = 0.65) -> None:
        super().__init__()
        self.temperature1 = temperature1
        self.temperature2 = temperature2
        self.residual_gain = residual_gain
        self.fuse_average = fuse_average
        self.fullres_mode = fullres_mode
        self.tile_w = tile_w
        self.overlap = overlap

        # Band controls
        self.use_band = use_band
        self.band_pad = band_pad
        self.band_min_width = band_min_width
        self.band_conf = band_conf

        # Hourglasses
        self.hg1 = Hourglass3D(in_ch=in_ch, base=base, depth=depth, use_ckpt=use_ckpt)
        self.inject = nn.Conv3d(1, in_ch, kernel_size=1, bias=False)
        self.hg2 = Hourglass3D(in_ch=in_ch, base=base, depth=depth, use_ckpt=use_ckpt)

        # Full-res upsampler (for "disp-jbu")
        self.fullres_up = GuidedDispUpsampler(use_featup=use_featup) if make_fullres else None

    def forward(self,
                C: torch.Tensor,
                *,
                orig_hw: Optional[Tuple[int, int]] = None,
                guidance: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Aggregate cost and (optionally) output original-size disparity.

        Args:
          C:        [B, C, D, H4, W4] cost (lower=better).
          orig_hw:  (H,W) original size. If provided, returns 'disp_full'.
          guidance: [B,3,H,W] for "disp-jbu" upsampling.

        Returns:
          dict with:
            'disp_lr':        [B,1,H4,W4] disparity (index units, already +d0)
            'disp_full':      [B,1,H,W]   original-size disparity (pixels)
            'prob':           [B,Ds,H4,W4] sub-band probability
            'cost':           [B,Ds,H4,W4] sub-band scalar cost
            'aux':            {'disp1','prob1','cost1'}
            'band':           (d0,d1)
            'band_offset_px': float, d0 mapped to full-res pixel units
        """
        # ------- Stage-1 -------
        feat1, cost1 = self.hg1(C)                                     # cost1: [B,D,H4,W4]
        disp1, prob1 = _soft_argmin(cost1, temp=self.temperature1)     # prob1: [B,D,H4,W4]

        # ------- Band selection from prob1 -------
        if self.use_band:
            with torch.no_grad():
                d0, d1 = _band_from_prob(prob1,
                                         min_conf=self.band_conf,
                                         pad=self.band_pad,
                                         min_width=self.band_min_width)
            C_sub = C[:, :, d0:d1 + 1, :, :]                         # [B,C,Ds,H4,W4]
            inj_all = self.inject(cost1.unsqueeze(1))                  # [B,C,D,H4,W4]
            inj_sub = inj_all[:, :, d0:d1 + 1, :, :]
            C2 = C_sub + self.residual_gain * inj_sub                  # HG2 仅看子带
        else:
            d0, d1 = 0, C.shape[2] - 1
            C2 = C + self.residual_gain * self.inject(cost1.unsqueeze(1))

        # ------- Stage-2 -------
        _, cost2_sub = self.hg2(C2)                                    # [B,Ds,H4,W4]
        if self.fuse_average:
            cost1_sub = cost1[:, d0:d1 + 1, :, :]
            cost_sub = 0.5 * (cost1_sub + cost2_sub)
        else:
            cost_sub = cost2_sub

        # soft-argmin on subrange -> 加回 d0（索引单位）
        disp_lr_local, prob_sub = _soft_argmin(cost_sub, temp=self.temperature2)  # [B,1,H4,W4], [B,Ds,H4,W4]
        disp_lr = disp_lr_local + float(d0)                                       # [B,1,H4,W4]

        out: Dict[str, torch.Tensor] = {
            "disp_lr": disp_lr,
            "prob": prob_sub,
            "cost": cost_sub,
            "aux": {"disp1": disp1, "prob1": prob1, "cost1": cost1},
            "band": (int(d0), int(d1)),
        }

        # ------- Full-res branch -------
        if orig_hw is not None:
            H4, W4 = disp_lr.shape[-2:]
            H, W = orig_hw
            scale_x = float(W) / float(W4)
            band_offset_px = float(d0) * scale_x  # d0 映射到原图像素单位

            if self.fullres_mode == "prob-dhw":
                out["disp_full"] = _fullres_from_prob_dhw(
                    P_lr=prob_sub, orig_hw=orig_hw,
                    d_scale=None, band_offset=band_offset_px, amp=not self.training
                )
            elif self.fullres_mode == "prob-dhw-tiled":
                out["disp_full"] = _fullres_from_prob_dhw_tiled(
                    P_lr=prob_sub, orig_hw=orig_hw,
                    d_scale=None, band_offset=band_offset_px,
                    tile_w=self.tile_w, overlap=self.overlap, amp=not self.training
                )
            else:  # "disp-jbu": 直接上采样 disp_lr（已含 d0），再做单位缩放
                if self.fullres_up is not None:
                    out["disp_full"] = self.fullres_up(disp_lr, guidance, orig_hw)
                else:
                    out["disp_full"] = F.interpolate(
                        disp_lr, size=orig_hw, mode="bilinear", align_corners=True
                    ) * scale_x

            out["band_offset_px"] = torch.tensor(band_offset_px, device=disp_lr.device)

        return out
