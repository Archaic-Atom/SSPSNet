# -*- coding: utf-8 -*-
"""Sparse prompt to dense full-resolution disparity.

This module converts a probability volume into high-confidence sparse anchors
and refines a base full-resolution disparity via edge-aware Laplacian propagation.

Two modes are supported to obtain the full-res base:
  - "prob-dhw":  Upsample P in D/H/W to (D*,H,W), renormalize, soft-argmin.
  - "disp-jbu":  Soft-argmin at H4,W4, then JBU or bilinear upsample to H,W.

Anchors can be extracted either from the full-res probability ("prob-dhw")
or from the low-res probability with coordinate scaling ("lr").

Outputs are in ORIGINAL pixel units. If your cost/prob used a disparity band
[dmin:dmax] at H4,W4, pass band_offset = dmin * (W/W4).

Google-style, pure PyTorch.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple, List
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional FeatUp JBU
try:
    from featup.upsamplers import JBULearnedRange
    _HAS_FEATUP = True
except Exception:
    _HAS_FEATUP = False


# --------------------------------------------------------------------------
# Basic utilities
# --------------------------------------------------------------------------
def soft_argmin_from_prob(P: torch.Tensor) -> torch.Tensor:
    """Soft-argmin disparity from probability volume.

    Args:
      P: [B, D, H, W], sum_D = 1
    Returns:
      disp: [B, 1, H, W] in disparity index units (0..D-1).
    """
    B, D, H, W = P.shape
    d_vals = torch.arange(D, device=P.device, dtype=P.dtype).view(1, D, 1, 1)
    return (P * d_vals).sum(1, keepdim=True)


def _local_max2d(score: torch.Tensor, k: int = 3) -> torch.Tensor:
    """2D NMS for [B,1,H,W] (ties kept)."""
    pad = k // 2
    pool = F.max_pool2d(score, kernel_size=k, stride=1, padding=pad)
    return score >= pool


def _interp3d_prob(P_lr: torch.Tensor, size_dhw: Tuple[int, int, int], amp: bool = True) -> torch.Tensor:
    """Trilinear upsample prob to target (D,H,W) and renormalize on D."""
    x = P_lr.unsqueeze(1)  # [B,1,Dlr,Hlr,Wlr]
    ctx = torch.autocast("cuda", torch.bfloat16) if (amp and torch.cuda.is_available()) else nullcontext()
    with ctx:
        P_hr = F.interpolate(x, size=size_dhw, mode="trilinear", align_corners=True).squeeze(1)
        P_hr = P_hr / (P_hr.sum(1, keepdim=True) + 1e-6)
    return P_hr


# --------------------------------------------------------------------------
# Upsamplers (JBU fallback to bilinear)
# --------------------------------------------------------------------------
class GuidedDispUpsampler(nn.Module):
    """Edge-aware disparity upsampler (H4,W4 -> H,W)."""

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


# --------------------------------------------------------------------------
# Sparse anchors (USPF)
# --------------------------------------------------------------------------
def extract_sparse_anchors(
    P: torch.Tensor,
    *,
    min_conf: float = 0.65,
    nms_ks: int = 3,
    topk: Optional[int] = 8000,
    edge: Optional[torch.Tensor] = None,   # [B,1,H,W] in [0,1]
    occ: Optional[torch.Tensor] = None,    # [B,1,H,W] in [0,1]
) -> Dict[str, torch.Tensor]:
    """High-confidence anchors from a probability volume (full-res or low-res).

    Args:
      P: [B, D, H, W] probability.
    Returns:
      dict with 'coords':[N,4](b,y,x,d), 'conf':[N], 'disp':[B,1,H,W].
    """
    B, D, H, W = P.shape
    disp = soft_argmin_from_prob(P)                         # [B,1,H,W]
    pmax, d_hat = P.max(1)                                  # [B,H,W], [B,H,W]
    top2 = torch.topk(P, k=min(2, D), dim=1).values
    p2 = top2[:, 1] if D >= 2 else torch.zeros_like(pmax)
    psr = pmax / (p2 + 1e-6)
    conf = pmax * psr.tanh()                                # (0,1) approx

    if edge is not None:
        conf = conf * torch.exp(-2.0 * edge.squeeze(1))
    if occ is not None:
        conf = conf * (1.0 - occ.squeeze(1)).clamp_min(0.0)

    keep = (conf.unsqueeze(1) >= min_conf) & _local_max2d(conf.unsqueeze(1), k=nms_ks)
    coords_list, conf_list = [], []
    for b in range(B):
        ys, xs = torch.nonzero(keep[b, 0], as_tuple=True)
        if ys.numel() == 0:
            continue
        scores = conf[b, ys, xs]
        if topk is not None and scores.numel() > topk:
            sel = torch.topk(scores, k=topk).indices
            ys, xs, scores = ys[sel], xs[sel], scores[sel]
        ds = d_hat[b, ys, xs]
        bb = torch.full_like(ys, b)
        coords_list.append(torch.stack([bb, ys, xs, ds], dim=1))
        conf_list.append(scores)

    if len(coords_list) == 0:
        coords = torch.zeros(0, 4, dtype=torch.long, device=P.device)
        confid = torch.zeros(0, dtype=P.dtype, device=P.device)
    else:
        coords = torch.cat(coords_list, dim=0)
        confid = torch.cat(conf_list, dim=0)

    return {"coords": coords, "conf": confid, "disp": disp}


def rasterize_dense_uspf(
    coords: torch.Tensor,
    conf: torch.Tensor,
    shape_hw: Tuple[int, int],
    D: int,
    conf_thresh: float = 0.65,
) -> Dict[str, torch.Tensor]:
    """Rasterize sparse coords into dense maps at (H,W): d_hat / c_hat / valid."""
    H, W = shape_hw
    if coords.numel() == 0:
        dev = conf.device
        z = torch.zeros((1, 1, H, W), dtype=torch.float32, device=dev)
        return {"d_hat": z, "c_hat": z, "valid": z.bool()}

    b, y, x, d = coords[:, 0].long(), coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()
    B = int(b.max().item()) + 1
    dev = conf.device
    d_map = torch.zeros((B, 1, H, W), dtype=torch.float32, device=dev)
    c_map = torch.zeros((B, 1, H, W), dtype=torch.float32, device=dev)
    v_map = torch.zeros((B, 1, H, W), dtype=torch.bool, device=dev)

    d = d.clamp(0, D - 1)
    c = conf.clamp(0, 1)
    idx = b * (H * W) + y.clamp(0, H - 1) * W + x.clamp(0, W - 1)
    d_map.view(-1).index_put_((idx,), d.float(), accumulate=False)
    c_map.view(-1).index_put_((idx,), c, accumulate=False)
    v_map.view(-1).index_put_((idx,), (c >= conf_thresh), accumulate=False)
    return {"d_hat": d_map, "c_hat": c_map, "valid": v_map}


# --------------------------------------------------------------------------
# Edge-aware Laplacian propagation at FULL resolution
# --------------------------------------------------------------------------
def refine_with_sparse_anchors_full(
    disp_init: torch.Tensor,      # [B,1,H,W] base (original units)
    d_hat: torch.Tensor,          # [B,1,H,W] anchor disparity indices (in same units as disp_init)
    c_hat: torch.Tensor,          # [B,1,H,W] confidence 0..1
    valid: torch.Tensor,          # [B,1,H,W] bool
    edge: Optional[torch.Tensor] = None,  # [B,1,H,W] 0..1
    occ: Optional[torch.Tensor] = None,   # [B,1,H,W] 0..1
    iters: int = 30,
    lam: float = 0.8,
    lambda_e: float = 2.0,
) -> torch.Tensor:
    """Refine full-res disparity with edge-aware diffusion of residual."""
    if edge is None:
        edge = torch.zeros_like(disp_init)
    if occ is None:
        occ = torch.zeros_like(disp_init)

    delta = (d_hat - disp_init).detach()
    m = (c_hat * valid.float()).detach()
    R = torch.zeros_like(disp_init)

    def _shift(x, dy, dx):
        return F.pad(x, (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0)), mode="replicate")[...,
                                                                                              max(-dy, 0):x.shape[-2] + max(-dy, 0), max(-dx, 0):x.shape[-1] + max(-dx, 0)]

    e = edge
    o = occ
    e_up = _shift(e, -1, 0)
    o_up = _shift(o, -1, 0)
    e_down = _shift(e, 1, 0)
    o_down = _shift(o, 1, 0)
    e_left = _shift(e, 0, -1)
    o_left = _shift(o, 0, -1)
    e_right = _shift(e, 0, 1)
    o_right = _shift(o, 0, 1)

    w_up = torch.exp(-lambda_e * (e + e_up) / 2) * (1 - o) * (1 - o_up)
    w_down = torch.exp(-lambda_e * (e + e_down) / 2) * (1 - o) * (1 - o_down)
    w_left = torch.exp(-lambda_e * (e + e_left) / 2) * (1 - o) * (1 - o_left)
    w_right = torch.exp(-lambda_e * (e + e_right) / 2) * (1 - o) * (1 - o_right)
    w_sum = w_up + w_down + w_left + w_right + 1e-6

    for _ in range(iters):
        Ru = _shift(R, -1, 0)
        Rd = _shift(R, 1, 0)
        Rl = _shift(R, 0, -1)
        Rr = _shift(R, 0, 1)
        nbr = w_up * Ru + w_down * Rd + w_left * Rl + w_right * Rr
        R = (m * delta + lam * nbr) / (m + lam * w_sum + 1e-6)

    return (disp_init + R).clamp_min(0.0)


# --------------------------------------------------------------------------
# Public API (full-res)
# --------------------------------------------------------------------------
class SparsePromptFull(nn.Module):
    """Sparse-prompt head that returns FULL-RES dense disparity.

    Args:
      mode: "prob-dhw" | "disp-jbu"
      anchor_mode: "auto" | "prob-dhw" | "lr"
        - "auto": use "prob-dhw" if mode=="prob-dhw", else "lr".
      use_featup: whether to use JBU in "disp-jbu"
    """

    def __init__(self, mode: str = "prob-dhw", anchor_mode: str = "auto", use_featup: bool = True) -> None:
        super().__init__()
        self.mode = mode
        self.anchor_mode = anchor_mode
        self.upsampler = GuidedDispUpsampler(use_featup=use_featup)

    @torch.no_grad()
    def _fullres_prob(self, P_lr: torch.Tensor, orig_hw: Tuple[int, int], d_scale: Optional[float], amp: bool) -> torch.Tensor:
        """Get full-res probability by 3D upsampling."""
        B, Dlr, Hlr, Wlr = P_lr.shape
        H, W = orig_hw
        scale_x = float(W) / float(Wlr)
        d_scale = scale_x if d_scale is None else d_scale
        Dhr = max(1, int(round(Dlr * d_scale)))
        return _interp3d_prob(P_lr, (Dhr, H, W), amp=amp)

    def forward(
        self,
        P_lr: torch.Tensor,                  # [B,Dlr,H4,W4]
        orig_hw: Tuple[int, int],            # (H,W)
        *,
        guidance: Optional[torch.Tensor] = None,  # [B,3,H,W] for JBU (disp-jbu)
        edge: Optional[torch.Tensor] = None,      # [B,1,H,W] optional
        occ: Optional[torch.Tensor] = None,       # [B,1,H,W] optional
        min_conf: float = 0.65,
        nms_ks: int = 3,
        topk: Optional[int] = 8000,
        iters: int = 30,
        lam: float = 0.8,
        lambda_e: float = 2.0,
        band_offset: float = 0.0,
        d_scale: Optional[float] = None,     # None -> use W/W4
        amp: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Return dense full-res disparity guided by sparse prompts.

        Returns:
          dict with:
            'disp_full': [B,1,H,W] final disparity
            'disp_init': [B,1,H,W] base (before refinement)
            'anchors':   {'coords':[N,4], 'conf':[N]} at FULL-RES coords
            'uspf':      {'d_hat','c_hat','valid'} dense maps [B,1,H,W]
        """
        B, Dlr, H4, W4 = P_lr.shape
        H, W = orig_hw
        scale_x = float(W) / float(W4)
        mode = self.mode
        a_mode = (self.anchor_mode if self.anchor_mode != "auto"
                  else ("prob-dhw" if mode == "prob-dhw" else "lr"))

        # (1) Get base full-res disparity AND a probability to extract anchors from
        if mode == "prob-dhw":
            # Full-res probability & base disparity
            P_full = self._fullres_prob(P_lr, orig_hw, d_scale=d_scale, amp=amp)      # [B,Dhr,H,W]
            disp_init = soft_argmin_from_prob(P_full) + float(band_offset)            # [B,1,H,W]
            P_for_anchor = P_full                                                     # extract anchors here
        else:  # "disp-jbu"
            # Low-res disparity then upsample (units -> original pixels)
            disp_lr = soft_argmin_from_prob(P_lr)                                     # [B,1,H4,W4]
            disp_init = self.upsampler(disp_lr, guidance, orig_hw) + float(band_offset)
            if a_mode == "prob-dhw":
                P_for_anchor = self._fullres_prob(P_lr, orig_hw, d_scale=d_scale, amp=amp)
            else:
                P_for_anchor = None  # we'll extract from low-res and scale coords

        # (2) Extract sparse anchors
        if P_for_anchor is not None:
            # Full-res anchors directly
            anchors = extract_sparse_anchors(P_for_anchor, min_conf=min_conf, nms_ks=nms_ks, topk=topk, edge=edge, occ=occ)
            coords_full, conf_full = anchors["coords"], anchors["conf"]
            Dhr = (P_for_anchor.shape[1] if mode == "prob-dhw" or a_mode == "prob-dhw" else Dlr)
        else:
            # Low-res anchors, then scale to full-res
            anchors_lr = extract_sparse_anchors(P_lr, min_conf=min_conf, nms_ks=nms_ks, topk=topk, edge=None, occ=None)
            coords_lr, conf_lr = anchors_lr["coords"], anchors_lr["conf"]
            if coords_lr.numel() == 0:
                coords_full = coords_lr.new_zeros((0, 4), dtype=torch.long)
                conf_full = conf_lr.new_zeros((0,), dtype=conf_lr.dtype)
            else:
                b = coords_lr[:, 0]
                y = (coords_lr[:, 1].float() * (H / float(H4))).round().long().clamp(0, H - 1)
                x = (coords_lr[:, 2].float() * (W / float(W4))).round().long().clamp(0, W - 1)
                d = (coords_lr[:, 3].float() * scale_x + float(band_offset)).round().long()
                coords_full = torch.stack([b.long(), y, x, d.clamp(min=0)], dim=1)
                conf_full = conf_lr
            Dhr = int(round(Dlr * (scale_x if d_scale is None else d_scale)))

        # (3) Rasterize USPF at FULL resolution
        uspf = rasterize_dense_uspf(coords_full, conf_full, (H, W), D=Dhr, conf_thresh=min_conf)

        # (4) Edge-aware refinement
        disp_final = refine_with_sparse_anchors_full(
            disp_init=disp_init,
            d_hat=uspf["d_hat"],
            c_hat=uspf["c_hat"],
            valid=uspf["valid"],
            edge=edge, occ=occ,
            iters=iters, lam=lam, lambda_e=lambda_e
        )

        return {
            "disp_full": disp_final,
            "disp_init": disp_init,
            "anchors": {"coords": coords_full, "conf": conf_full},
            "uspf": uspf,
        }
