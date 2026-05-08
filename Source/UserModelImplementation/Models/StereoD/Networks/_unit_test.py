# -*- coding: utf-8 -*-
import os
import sys
import torch
from torch import nn
from torchsummary import summary
from einops import rearrange

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
    from .model import StereoA
    from ._prompt import Prompt
    from ._offset import Refinement, RefinementLayer, MLP
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
    from model import StereoA
    from _prompt import Prompt
    from _offset import Refinement, RefinementLayer, MLP


class UnitTest(object):
    def __init__(self):
        super().__init__()

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
        # refine_head = MLP(infer_embed_dim, infer_embed_dim, image_scaling_ratio**2, 3)
        refine_head = MLP(infer_embed_dim, infer_embed_dim, 8, 3)

        return refinement, refine_head

    def _depth_test(self) -> None:
        g = torch.rand(2, 64, 448, 224).cuda()
        left_img = torch.rand(3, 3, 448, 224).cuda()
        right_img = torch.rand(3, 3, 448, 224).cuda()
        left_img_gw = torch.rand(3, 32, 448, 224).cuda()
        right_img_gw = torch.rand(3, 32, 448, 224).cuda()
        sparse_depth = torch.rand(2, 1, 448, 224).cuda()
        blur_depth = torch.rand(2, 1, 448, 224).cuda()
        depth = torch.rand(3, 448, 224).cuda()

        #model = Prompt(8).cuda()
        model = StereoA(3, 0, 196, 'dinov2', False).cuda()
        #refine, refine_head = self._refine_layers(1)
        # refine = refine.cuda()
        # refine_head = refine_head.cuda()

        with torch.no_grad():
            # tgt = refine(depth, left_img, right_img, left_img_gw, right_img_gw)
            # disp_delta = refine_head(tgt)
            # disp_delta = rearrange(disp_delta, 'a (b h w) c -> (a b) c h w ', h=448, w=224)
            # print(disp_delta.shape)
            #res = model(g, blur_depth, sparse_depth, (448, 224))
            res = model(left_img, right_img)

    def exec(self, args: object) -> None:
        self._depth_test()


def main() -> None:
    unit_test = UnitTest()
    unit_test.exec(None)


if __name__ == '__main__':
    main()
