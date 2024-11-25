# -*- coding: utf-8 -*-
import torch
from torch import nn
from timm.models.layers import trunc_normal_
from einops import rearrange
from torch.nn import functional as F

try:
    from ._nmp import (
        MLP,
        InferenceLayer,
        Inference,
        RefinementLayer,
        Refinement,
    )
except ImportError:
    from _nmp import (
        MLP,
        InferenceLayer,
        Inference,
        RefinementLayer,
        Refinement,
    )


class NMRF(nn.Module):
    def __init__(self, in_channels: int, image_scaling_ratio: int = 14,
                 refinement_image_scaling_ratio: int = 2) -> None:
        super().__init__()
        self.concatconv, self.gw = self._concatconv(in_channels, 64), self._gw(in_channels, 256)
        self.inference, self.infer_head, self.infer_score_head = \
            self._infer_layers(image_scaling_ratio)
        self.image_scaling_ratio, self.refinement_image_scaling_ratio = \
            image_scaling_ratio, refinement_image_scaling_ratio
        self.amplify_size = self._amplify_size(
            in_channels, 64, image_scaling_ratio // refinement_image_scaling_ratio)
        self.refinement, self.refine_head = self._refine_layers(refinement_image_scaling_ratio)
        self.concatconv_refine, self.gw_refine = \
            self._concatconv(64, 8), self._gw(64, 32)
        self.apply(self._init_weights)

        # to keep track of which device the nn.Module is on
        self.register_buffer("device_indicator_tensor", torch.empty(0))

    def _concatconv(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 2, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 2, out_channels, 1, 1, 0, bias=False)
        )

    def _gw(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, 1, 1, 0, bias=False)
        )

    def _amplify_size(self, in_channels: int, out_channels: int,
                      image_scaling_ratio: int) -> nn.Module:
        return nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels, out_channels,
                               image_scaling_ratio,
                               image_scaling_ratio, 0, bias=False),
            nn.InstanceNorm2d(out_channels),
        )

    def _infer_layers(self,
                      image_scaling_ratio: int,
                      infer_embed_dim: int = 128,
                      num_infer_layers: int = 5,
                      mlp_ratio: float = 4,
                      window_size: int = 6,
                      infer_n_heads: int = 4,
                      activation: str = "gelu",
                      attn_drop: float = 0.,
                      proj_drop: float = 0.,
                      dropout: float = 0.,
                      drop_path: float = 0.,
                      normalize_before: bool = False,
                      return_intermediate=False) -> nn.Module:
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_infer_layers)]
        infer_layers = nn.ModuleList([
            InferenceLayer(
                infer_embed_dim, mlp_ratio=mlp_ratio, window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2, n_heads=infer_n_heads,
                activation=activation,
                attn_drop=attn_drop, proj_drop=proj_drop, drop_path=dpr[i], dropout=dropout,
                normalize_before=normalize_before
            )
            for i in range(num_infer_layers)]
        )
        inference = Inference(32, infer_embed_dim, layers=infer_layers,
                              norm=nn.LayerNorm(infer_embed_dim),
                              return_intermediate=return_intermediate)
        infer_head = MLP(infer_embed_dim, infer_embed_dim, image_scaling_ratio ** 2, 3)
        infer_score_head = nn.Linear(infer_embed_dim, image_scaling_ratio ** 2)
        return inference, infer_head, infer_score_head

    def _refine_layers(self,
                       image_scaling_ratio: int,
                       infer_embed_dim: int = 16,
                       num_refine_layers: int = 5,
                       mlp_ratio: float = 4,
                       refine_window_size: int = 6,
                       infer_n_heads: int = 4,
                       activation: str = "gelu",
                       attn_drop: float = 0.,
                       proj_drop: float = 0.,
                       dropout: float = 0.,
                       drop_path: float = 0.,
                       normalize_before: bool = False,
                       return_intermediate=False) -> nn.Module:
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_refine_layers)]
        refine_layers = nn.ModuleList([
            RefinementLayer(
                infer_embed_dim, mlp_ratio=mlp_ratio, window_size=refine_window_size,
                shift_size=0 if i % 2 == 0 else refine_window_size // 2, n_heads=infer_n_heads,
                activation=activation,
                attn_drop=attn_drop, proj_drop=proj_drop, drop_path=dpr[i], dropout=dropout,
                normalize_before=normalize_before,
            )
            for i in range(num_refine_layers)]
        )
        refine_norm = nn.LayerNorm(infer_embed_dim)
        refinement = Refinement(32, infer_embed_dim, layers=refine_layers, norm=refine_norm,
                                return_intermediate=return_intermediate)
        refine_head = MLP(infer_embed_dim, infer_embed_dim, image_scaling_ratio**2, 3)

        return refinement, refine_head

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.InstanceNorm2d)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def infer_forward(self, left_img: torch.Tensor,
                      right_img: torch.Tensor,
                      labels: list) -> torch.Tensor:
        left_feat = self.concatconv(left_img)
        right_feat = self.concatconv(right_img)

        left_feat_gw = self.gw(left_img)
        right_feat_gw = self.gw(left_img)

        labels_curr = labels[-1].detach()
        tgt = self.inference(labels_curr, left_feat, right_feat, left_feat_gw, right_feat_gw)
        disp_delta = self.infer_head(tgt)
        coarse_disp = F.relu(labels_curr[None].unsqueeze(-1) + disp_delta)
        mask = .25 * self.infer_score_head(tgt)
        _, _, ht, wd = left_feat.shape
        coarse_disp = rearrange(
            coarse_disp,
            'a (b h w) n (hs ws) -> a b (h hs) (w ws) n',
            h=ht, w=wd, hs=self.image_scaling_ratio).contiguous()
        mask = rearrange(mask, 'a (b h w) n (hs ws) -> a b (h hs) (w ws) n',
                         h=ht, w=wd, hs=14)
        return coarse_disp, mask

    def refine_forward(self, coarse_disp: torch.Tensor, mask: torch.Tensor,
                       left_img: torch.Tensor, right_img: torch.Tensor,
                       image_scaling_ratio: int = 14,
                       refinement_image_scaling_ratio: int = 2) -> torch.Tensor:
        _, indices = torch.max(mask[-1], dim=-1, keepdim=True)
        disp_curr = torch.gather(
            coarse_disp[-1], dim=-1, index=indices).squeeze(-1) * \
            image_scaling_ratio // refinement_image_scaling_ratio  # [B,H,W]
        disp_curr = rearrange(disp_curr,
                              'b (h hs) (w ws) -> b h w (hs ws)',
                              hs=refinement_image_scaling_ratio,
                              ws=refinement_image_scaling_ratio)
        disp_curr = torch.median(disp_curr, dim=-1, keepdim=False)[0]
        disp_curr = disp_curr.detach()
        left_feat = self.concatconv_refine(left_img)
        right_feat = self.concatconv_refine(right_img)

        left_feat_gw = self.gw_refine(left_img)
        right_feat_gw = self.gw_refine(right_img)
        tgt = self.refinement(disp_curr, left_feat, right_feat, left_feat_gw, right_feat_gw)

        disp_delta = self.refine_head(tgt)  # [num_aux_layers,BHW,4*4]
        _, _, ht, wd = left_feat.shape
        disp_delta = rearrange(disp_delta, 'a (b h w) p -> a b h w p', h=ht, w=wd)
        disp_pred = F.relu(disp_curr[None].unsqueeze(-1) + disp_delta)
        disp_pred = rearrange(disp_pred,
                              'a b h w (hs ws) -> a b (h hs) (w ws)',
                              hs=refinement_image_scaling_ratio).contiguous()
        return disp_pred

    def forward(self, left_img: torch.Tensor,
                right_img: torch.Tensor,
                labels: list) -> torch.Tensor:
        coarse_disp, mask = self.infer_forward(left_img, right_img, labels)
        left_img = self.amplify_size(left_img)
        right_img = self.amplify_size(right_img)
        disp_pred = self.refine_forward(
            coarse_disp, mask, left_img, right_img,
            self.image_scaling_ratio, self.refinement_image_scaling_ratio)
        return disp_pred, mask
