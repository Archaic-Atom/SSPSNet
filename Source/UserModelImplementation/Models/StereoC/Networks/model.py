# -*- coding: utf-8 -*-
from functools import partial

import os
import torch
from torch import nn
import torch.nn.functional as F


try:
    from ._backbone import get_dino_layers_id, create_backbone
    from ._upsampler import JBUStack
    from ._GCSF import GCSF
    from ._cost_volume import GDCCostVolumeChunkedBCDHW
    from ._feature_matching import HG3DPlus
    from ._sparse_prompt_head import SparsePromptFull
except ImportError:
    from _backbone import get_dino_layers_id, create_backbone
    from _upsampler import JBUStack
    from _GCSF import GCSF
    from _cost_volume import GDCCostVolumeChunkedBCDHW
    from _feature_matching import HG3DPlus
    from _sparse_prompt_head import SparsePromptFull


class StereoA(nn.Module):
    ROOT_PATH = '/data3/raozhibo/SAStereo/'
    WEIGHTS_PATH = 'Weights/dinov3/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth'

    def __init__(self, in_channles: int, start_disp: int = 1, disp_num: int = 192,
                 backbone_name: str = 'dinov3', confidence_level: float = 0.15) -> None:
        super().__init__()
        self.start_disp, self.disp_num = start_disp, disp_num
        self.in_channles = in_channles
        self.confidence_level = confidence_level
        self.backbone = create_backbone(backbone_name, 'dinov3_vith16plus',
                                        os.path.join(self.ROOT_PATH, self.WEIGHTS_PATH))
        self.resize_layers = self._get_resize_layers(1280)
        self.fusion = self._get_fusion_layers([256, 256, 256, 256])

        self.build_cost_volume = self._build_cost_volume(256)
        self.feature_matching = self._build_feature_matching(64)
        self.sparse_prompt_full = self._build_sparse_prompt()

    def _get_resize_layers(self, in_channles: int) -> nn.Module:
        return nn.ModuleList([
            JBUStack(feat_dim=in_channles, out_dim=256, mid_dim=256, radius=3)
            for i in range(len(get_dino_layers_id('dinov3_vith16plus')))
        ])

    def _get_fusion_layers(self, in_channles_list: list) -> nn.Module:
        return GCSF(in_chs=in_channles_list, out_ch=256,
                    reduction=4, norm=lambda c: nn.GroupNorm(32, c),   # 小 batch 推荐
                    act=lambda: nn.ReLU(inplace=True),
                    use_depthwise=True, dropout=0.0)

    def _feature_extraction_module_proc(
            self, left_img: torch.Tensor, right_img: torch.Tensor) -> tuple:
        with torch.no_grad():
            left_feat_list = self.backbone.get_intermediate_layers(
                left_img, n = get_dino_layers_id('dinov3_vith16plus'), reshape = True,
                return_class_token = False, norm = False)
            right_feat_list = self.backbone.get_intermediate_layers(
                right_img, n = get_dino_layers_id('dinov3_vith16plus'), reshape = True,
                return_class_token = False, norm = False)

        left_feat_list = [j(f, left_img) for j, f in zip(self.resize_layers, left_feat_list)]  # 每个→[B,256,H/4,W/4]
        right_feat_list = [j(f, right_img) for j, f in zip(self.resize_layers, right_feat_list)]

        left_feat = F.normalize(self.fusion(left_feat_list), dim=1)
        right_feat = F.normalize(self.fusion(right_feat_list), dim=1)

        return left_feat, right_feat

    def _build_cost_volume(self, in_channles: int) -> nn.Module:
        return GDCCostVolumeChunkedBCDHW(
            in_ch=in_channles, cost_ch=128, groups=64,
            chunk=12, use_band=True, use_ckpt=False)

    def _build_feature_matching(self, in_channles: int) -> nn.Module:
        return HG3DPlus(in_ch=in_channles, base=32, depth=2, use_ckpt=False,
                        make_fullres=True, fullres_mode="prob-dhw")

    def _build_sparse_prompt(self) -> nn.Module:
        return SparsePromptFull(mode="prob-dhw", anchor_mode="auto", use_featup=True)

    def forward(self, left_img: torch.Tensor, right_img: torch.Tensor) -> torch.Tensor:
        left_feat, right_feat = self._feature_extraction_module_proc(left_img, right_img)
        cost = self.build_cost_volume(left_feat, right_feat, self.disp_num)
        ret2 = self.feature_matching(cost, orig_hw=(left_img.shape[-2], left_img.shape[-1]))

        disp_full2 = ret2["disp_full"]

        print(disp_full2.shape)

        out = self.sparse_prompt_full(P_lr=ret2["prob"],
                                      orig_hw=(left_img.shape[-2], left_img.shape[-1]),
                                      guidance=left_img,       # prob-dhw 模式可为 None
                                      edge=None,               # 可选 [B,1,H,W]，没有就 None
                                      occ=None,                 # 可选 [B,1,H,W]，没有就 None
                                      min_conf=0.65, nms_ks=3, topk=8000,
                                      iters=30, lam=0.8, lambda_e=2.0
                                      # band_offset=0  # 若 cost 用了 dmin:dmax，传 dmin * (W/W4)
                                      )
        print(out["disp_full"].shape)


if __name__ == '__main__':
    model = StereoA(3).cuda()
    rand_tensor = torch.rand(1, 3, 640, 320).cuda()
    model.eval()
    model(rand_tensor, rand_tensor)
