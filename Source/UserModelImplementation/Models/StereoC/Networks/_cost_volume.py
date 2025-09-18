# cost_volume_bcdhw_train.py
# -*- coding: utf-8 -*-
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt


@torch.no_grad()
def _make_warp_grid(
    B: int, H: int, W: int, d0: int, d1: int, device, dtype, align_corners: bool = True
) -> torch.Tensor:
    """Build right-feature sampling grid for disparities [d0, d1).
    Returns: [B, H, W, Dc, 2] suitable for .view(B, H, W*Dc, 2) in grid_sample.
    """
    Dc = int(d1 - d0)
    # 基于 align_corners 的规范化坐标与位移尺度
    if align_corners:
        nx = torch.linspace(-1, 1, W, device=device, dtype=dtype)           # x ∈ [-1,1], len=W
        ny = torch.linspace(-1, 1, H, device=device, dtype=dtype)           # y ∈ [-1,1], len=H
        scale = 2.0 / max(W - 1, 1)                                         # 每 1 像素位移对应的归一化步长
    else:
        # 若你全程 align_corners=True，可不走这支；放这里仅为完整性
        nx = torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=device, dtype=dtype)
        ny = torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=device, dtype=dtype)
        scale = 2.0 / max(W, 1)

    # 形状设计：把 Dc 放在**最后一维**，保证与 x 的最后一维(=1)能广播
    x = nx.view(1, 1, 1, W, 1)                                              # [1,1,1,W,1]
    y = ny.view(1, 1, H, 1, 1)                                              # [1,1,H,1,1]
    disp = torch.arange(d0, d1, device=device, dtype=dtype).view(1, 1, 1, 1, Dc)  # [1,1,1,1,Dc]

    x_shift = x - disp * scale                                              # [1,1,1,W,Dc]
    # 广播到 [B,H,W,Dc]
    x_shift = x_shift.expand(B, 1, H, W, Dc).squeeze(1)                     # [B,H,W,Dc]
    y_grid = y.expand(B, 1, H, W, Dc).squeeze(1)                           # [B,H,W,Dc]

    grid = torch.stack((x_shift, y_grid), dim=-1)                           # [B,H,W,Dc,2]
    return grid


@torch.no_grad()
def compute_disparity_band_from_anchors(d_hat: torch.Tensor, valid: torch.Tensor, Dp: int,
                                        pad: int = 4, q=(0.02, 0.98)) -> Tuple[int, int]:
    B = d_hat.shape[0]
    dmins, dmaxs = [], []
    for b in range(B):
        dh = d_hat[b, 0][valid[b, 0]]
        if dh.numel() == 0:
            dmins.append(0)
            dmaxs.append(Dp - 1)
            continue
        lo, hi = torch.quantile(dh, torch.tensor(q, device=dh.device))
        d0 = int(max(0, math.floor(lo.item()) - pad))
        d1 = int(min(Dp - 1, math.ceil(hi.item()) + pad))
        if d0 > d1:
            d0, d1 = d1, d0
        dmins.append(d0)
        dmaxs.append(d1)
    return min(dmins), max(dmaxs)


class GDCCostVolumeChunkedBCDHW(nn.Module):
    """[B, C(=groups), D, H, W] 输出的省显存 GDCC（训练友好版）."""

    def __init__(self, in_ch: int, cost_ch: int = 96, groups: int = 8, tau: float = 0.07,
                 chunk: int = 12, use_band: bool = True, align_corners: bool = True,
                 padding_mode: str = "border", use_ckpt: bool = True, amp_dtype=torch.float16) -> None:
        super().__init__()
        assert cost_ch % groups == 0
        self.cost_ch, self.groups, self.tau = cost_ch, groups, tau
        self.chunk, self.use_band = chunk, use_band
        self.align_corners, self.padding_mode = align_corners, padding_mode
        self.use_ckpt, self.amp_dtype = use_ckpt, amp_dtype
        self.pL = nn.Conv2d(in_ch, cost_ch, 1, bias=False)
        self.pR = nn.Conv2d(in_ch, cost_ch, 1, bias=False)

    def _chunk_cost(self, FLg, FR, d0: int, d1: int, Dp_full: int, H: int, W: int):
        """返回 [B,G,H,W,Dc]（供 checkpoint 复算），只依赖张量输入。"""
        B, G, Cg, _, _ = FLg.shape
        Dc = d1 - d0
        grid = _make_warp_grid(B, H, W, d0, d1, FR.device, FR.dtype)  # [B,H,W,Dc,2]
        FRs = F.grid_sample(
            FR, grid.view(B, H, W * Dc, 2),
            mode="bilinear", padding_mode=self.padding_mode, align_corners=self.align_corners
        ).view(B, self.cost_ch, H, W, Dc).view(B, G, Cg, H, W, Dc)
        sim = (FLg.unsqueeze(-1) * FRs).sum(dim=2) / (Cg ** 0.5)  # [B,G,H,W,Dc]
        dvec = torch.arange(d0, d1, device=FR.device, dtype=FR.dtype)[None, None, None, None, :]
        psi = torch.sin(2 * math.pi * dvec / max(Dp_full, 1))
        return -(sim + psi) / self.tau

    def forward(self, F_L: torch.Tensor, F_R: torch.Tensor, D_orig: int,
                d_hat: Optional[torch.Tensor] = None, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, _, H, W = F_L.shape
        Dp_full = int(math.ceil(D_orig / 4.0))
        if self.use_band and (d_hat is not None) and (valid is not None):
            dmin, dmax = compute_disparity_band_from_anchors(d_hat, valid, Dp_full, pad=4, q=(0.02, 0.98))
        else:
            dmin, dmax = 0, Dp_full - 1
        Dp = dmax - dmin + 1

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True):
            FL = F.normalize(self.pL(F_L), dim=1)
            FR = F.normalize(self.pR(F_R), dim=1)

        G, Cg = self.groups, self.cost_ch // self.groups
        FLg = FL.view(B, G, Cg, H, W)

        C_bghwd = FL.new_zeros((B, G, H, W, Dp), dtype=FL.dtype)
        d = dmin
        while d <= dmax:
            d0, d1 = d, min(d + self.chunk, dmax + 1)
            if self.use_ckpt and self.training:
                # checkpoint 复算该 chunk，显存大幅下降（计算量↑）
                cost_chunk = ckpt(self._chunk_cost, FLg, FR, torch.tensor(d0, device=FL.device),
                                  torch.tensor(d1, device=FL.device), torch.tensor(Dp_full, device=FL.device),
                                  torch.tensor(H, device=FL.device), torch.tensor(W, device=FL.device))
            else:
                cost_chunk = self._chunk_cost(FLg, FR, d0, d1, Dp_full, H, W)
            C_bghwd[..., (d0 - dmin):(d1 - dmin)] = cost_chunk
            # 及时释放临时张量引用
            del cost_chunk
            d = d1

        return C_bghwd.permute(0, 1, 4, 2, 3).contiguous()  # [B,G,D,H,W]
