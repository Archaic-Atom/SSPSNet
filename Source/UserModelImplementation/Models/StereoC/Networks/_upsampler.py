# jbu_stack.py
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from featup.upsamplers import JBULearnedRange


class JBUStack(nn.Module):
    """Two-stage JBU upsampling from H/16 -> H/8 -> H/4.

    This module applies *learned joint bilateral upsampling* (FeatUp: JBULearnedRange)
    twice (×2 then ×2) to lift ViT features from 1/16 resolution to 1/4. It is designed
    for stereo pipelines: **use left RGB to guide left features, right RGB to guide
    right features**, and (recommended) **share the same JBUStack weights** across L/R
    for statistical symmetry.

    The block also includes an optional pre-projection to reduce channels before JBU
    (saves memory/compute), a light FixUp-style residual, and an optional normalization
    to stabilize training.

    Example:
      >>> jbu = JBUStack(feat_dim=384, out_dim=256, mid_dim=256, radius=3)
      >>> F_L_h16 = torch.randn(2, 384, 48, 80)   # H/16, W/16
      >>> img_L   = torch.randn(2, 3, 768, 1280)  # H, W (0~1 normalized is recommended)
      >>> F_L_h4  = jbu(F_L_h16, img_L)           # -> [2, 256, 192, 320]  (H/4, W/4)

    Args:
      feat_dim: Number of channels of the input low-res feature (H/16).
      out_dim: Number of channels of the output feature (H/4).
      mid_dim: Optional channel count used *inside* JBU (pre-projection). If None,
        uses `feat_dim`. Set e.g. 256 to reduce cost when `feat_dim` is large.
      radius: JBU range kernel radius. 3 is a safe default from FeatUp.
      dropout: Dropout rate in the lightweight FixUp projection.
      use_norm: Whether to apply a light normalization after FixUp (GroupNorm).
        Set False if you prefer to keep the distribution untouched.

    Returns:
      A tensor of shape `[B, out_dim, H/4, W/4]`.

    Notes:
      * Guidance should match the *target* spatial size at each JBU step; we resize
        internally with bilinear interpolation (no align_corners).
      * AMP: we cast guidance to the same dtype as `source` to avoid mixed-precision
        issues.
      * Weight sharing: instantiate **one** JBUStack and apply to left/right with
        different guidances to keep L/R statistics aligned.
    """

    def __init__(
        self,
        feat_dim: int,
        out_dim: int,
        mid_dim: Optional[int] = None,
        radius: int = 3,
        dropout: float = 0.2,
        use_norm: bool = True,
    ) -> None:
        super().__init__()

        self.in_dim = feat_dim if mid_dim is None else mid_dim

        # Optional pre-projection to reduce channels before JBU (saves compute/memory).
        self.pre_proj: Optional[nn.Conv2d]
        if mid_dim is not None and mid_dim != feat_dim:
            self.pre_proj = nn.Conv2d(feat_dim, self.in_dim, kernel_size=1, bias=False)
        else:
            self.pre_proj = None

        # Two ×2 JBU stages: H/16->H/8 and H/8->H/4.
        self.up1 = JBULearnedRange(guidance_dim=3, feat_dim=self.in_dim, key_dim=32, radius=radius)
        self.up2 = JBULearnedRange(guidance_dim=3, feat_dim=self.in_dim, key_dim=32, radius=radius)

        # Light FixUp-style residual to stabilize outputs (prevents ringing).
        self.fixup_proj = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(self.in_dim, self.in_dim, kernel_size=1, bias=False),
        )

        # Optional normalization to further stabilize training.
        self.norm = nn.GroupNorm(num_groups=min(32, max(1, self.in_dim // 8)),
                                 num_channels=self.in_dim) if use_norm else None

        # Final projection to the requested output channels.
        self.proj = nn.Conv2d(self.in_dim, out_dim, kernel_size=1, bias=False)

    @staticmethod
    def _resize_guidance(guidance: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Resize guidance image/tensor to target (h, w) with bilinear interpolation.

        Args:
          guidance: Guidance tensor `[B, 3, H, W]` (preferably 0~1 normalized).
          h: Target height.
          w: Target width.

        Returns:
          Resized guidance tensor `[B, 3, h, w]`.
        """
        return F.interpolate(guidance, size=(h, w), mode="bilinear", align_corners=False)

    def _upsample_once(
        self,
        source: torch.Tensor,
        guidance: torch.Tensor,
        up: JBULearnedRange,
    ) -> torch.Tensor:
        """Apply one ×2 JBU step with proper guidance resizing & dtype alignment.

        Args:
          source: Low-res feature `[B, C, h, w]`.
          guidance: Full-res RGB `[B, 3, H, W]`.
          up: A JBULearnedRange instance.

        Returns:
          Upsampled feature `[B, C, 2h, 2w]`.
        """
        guidance = guidance.to(dtype=source.dtype)
        _, _, h, w = source.shape
        g = self._resize_guidance(guidance, h * 2, w * 2)
        return up(source, g)

    def forward(self, source: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        """Forward pass: H/16 -> H/8 -> H/4 via two JBU stages.

        Args:
          source: Input feature at 1/16 resolution `[B, feat_dim, H/16, W/16]`.
          guidance: RGB guidance at full resolution `[B, 3, H, W]`.

        Returns:
          Output feature at 1/4 resolution `[B, out_dim, H/4, W/4]`.
        """
        x = source
        if self.pre_proj is not None:
            x = self.pre_proj(x)

        # H/16 -> H/8 -> H/4
        x = self._upsample_once(x, guidance, self.up1)
        x = self._upsample_once(x, guidance, self.up2)

        # FixUp residual + (optional) normalization
        x = x + 0.1 * self.fixup_proj(x)
        if self.norm is not None:
            x = self.norm(x)

        return self.proj(x)
