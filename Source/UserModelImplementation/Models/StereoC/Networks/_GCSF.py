# gcsf.py
# -*- coding: utf-8 -*-
from typing import List, Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCSF(nn.Module):
    """Gated Cross-Scale Fusion for four H/4 feature maps.

    This module fuses four upsampled feature maps (e.g., from different backbone
    layers) at the same spatial resolution (typically H/4, W/4). It first aligns
    channels per-branch, extracts lightweight local context via depthwise conv,
    and then computes **spatial gates** and a **global SE gate**. The final
    per-branch weights are obtained by a softmax over the sum of spatial and
    global logits, and used to blend branches. A final 1×1 projection produces
    the fused output.

    The design is intentionally light and memory-friendly for stereo pipelines.

    Example:
      >>> B, H4, W4 = 2, 192, 320
      >>> F1 = torch.randn(B, 256, H4, W4)
      >>> F2 = torch.randn(B, 256, H4, W4)
      >>> F3 = torch.randn(B, 256, H4, W4)
      >>> F4 = torch.randn(B, 256, H4, W4)
      >>> gcsf = GCSF(in_chs=[256, 256, 256, 256], out_ch=256, reduction=4)
      >>> y = gcsf([F1, F2, F3, F4])
      >>> y.shape
      torch.Size([2, 256, 192, 320])

    Args:
      in_chs: A list of four integers for input channels of each branch.
      out_ch: Output channel dimension after fusion (default: 256).
      reduction: Reduction ratio used in the global SE gate (default: 4).
      norm: Normalization layer factory. If None, no normalization is applied.
        Typical choices: `nn.BatchNorm2d`, `lambda c: nn.GroupNorm(32, c)`.
      act: Activation layer factory applied after per-branch projection
        (default: `nn.ReLU(inplace=True)`).
      use_depthwise: Whether to apply a 3×3 depthwise conv per branch to mine
        local spatial cues for the spatial gate (default: True).
      dropout: Dropout probability in the final 1×1 projection (default: 0.0).

    Returns:
      A fused tensor of shape `[B, out_ch, H, W]`.

    Raises:
      AssertionError: If `len(in_chs) != 4` or inputs have mismatched spatial size.

    Notes:
      * Inputs must share identical spatial resolution.
      * Spatial gate captures per-pixel branch preference; SE gate captures
        image-level branch preference. Softmax ensures weights across branches
        sum to 1 at every spatial location.
      * For very small batch sizes, consider GroupNorm instead of BatchNorm.
    """

    def __init__(
        self,
        in_chs: List[int],
        out_ch: int = 256,
        reduction: int = 4,
        norm: Optional[Callable[[int], nn.Module]] = nn.BatchNorm2d,
        act: Optional[Callable[[], nn.Module]] = lambda: nn.ReLU(inplace=True),
        use_depthwise: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert len(in_chs) == 4, "GCSF expects exactly four input branches."

        self.out_ch = out_ch
        self.use_depthwise = use_depthwise

        # Per-branch channel alignment and (optional) normalization/activation.
        self.proj = nn.ModuleList([nn.Conv2d(c, out_ch, kernel_size=1, bias=False) for c in in_chs])
        self.norm = nn.ModuleList([norm(out_ch) if norm is not None else nn.Identity() for _ in in_chs])
        self.act = nn.ModuleList([act() if act is not None else nn.Identity() for _ in in_chs])

        # Lightweight local context for spatial gate.
        if use_depthwise:
            self.dw = nn.ModuleList([
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch, bias=False)
                for _ in in_chs
            ])
            self.dw_act = act() if act is not None else nn.Identity()
            self.alpha_spatial = nn.ModuleList([nn.Conv2d(out_ch, 1, kernel_size=1, bias=True) for _ in in_chs])
        else:
            self.alpha_spatial = nn.ModuleList([nn.Conv2d(out_ch, 1, kernel_size=1, bias=True) for _ in in_chs])

        # Global SE gate (shared across spatial positions).
        hidden = max(out_ch // reduction, 16)
        self.se_fc1 = nn.Linear(out_ch, hidden, bias=False)
        self.se_fc2 = nn.Linear(hidden, 4, bias=False)

        # Final projection.
        self.out = nn.Sequential(
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
        )

    def forward(self, feats4: List[torch.Tensor]) -> torch.Tensor:
        """Fuse four features with gated cross-scale fusion.

        Args:
          feats4: List of four tensors, each `[B, C_i, H, W]`, same H/W.

        Returns:
          Fused feature `[B, out_ch, H, W]`.
        """
        assert len(feats4) == 4, "Provide exactly four feature maps."
        B, _, H, W = feats4[0].shape
        for i in range(1, 4):
            assert feats4[i].shape[-2:] == (H, W), "All inputs must share H/W."

        # 1) Per-branch alignment (+norm/act)
        xs = []
        for i, x in enumerate(feats4):
            x = self.proj[i](x)
            x = self.norm[i](x)
            x = self.act[i](x)
            xs.append(x)  # each: [B, out_ch, H, W]

        # 2) Spatial gate logits α_spatial (per-pixel, per-branch)
        alphas_sp = []
        for i, x in enumerate(xs):
            s = x
            if self.use_depthwise:
                s = self.dw_act(self.dw[i](s))
            alphas_sp.append(self.alpha_spatial[i](s))  # [B,1,H,W]
        A_spatial = torch.cat(alphas_sp, dim=1)  # [B,4,H,W]

        # 3) Global SE gate logits α_se (image-level, per-branch)
        x_sum = torch.stack(xs, dim=1).sum(dim=1)        # [B,out_ch,H,W]
        g = F.adaptive_avg_pool2d(x_sum, 1).view(B, -1)  # [B,out_ch]
        g = F.relu(self.se_fc1(g), inplace=True)
        A_se = self.se_fc2(g).view(B, 4, 1, 1)           # [B,4,1,1]

        # 4) Softmax over branch dimension to obtain normalized weights
        w = torch.softmax(A_spatial + A_se, dim=1)       # [B,4,H,W]

        # 5) Weighted sum over branches
        xs_stack = torch.stack(xs, dim=1)                # [B,4,out_ch,H,W]
        fused = (w.unsqueeze(2) * xs_stack).sum(dim=1)   # [B,out_ch,H,W]

        # 6) Final 1×1 projection
        return self.out(fused)
