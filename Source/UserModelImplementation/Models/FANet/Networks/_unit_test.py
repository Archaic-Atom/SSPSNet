# -*- coding: utf-8 -*-
import os
import sys
import torch

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
    from .models_mamba import vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
    from Libs.dinov2.dinov2.hub.depthers import dinov2_vitl14_dd
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
    from models_mamba import vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
    from Libs.dinov2.dinov2.hub.depthers import dinov2_vitl14_dd


class UnitTest(object):
    def __init__(self):
        super().__init__()

    def _mamba_test(self) -> None:
        model = vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2()
        model = model.cuda()
        left_img = torch.rand(1, 3, 224, 224).cuda()
        res = model(left_img, return_features=True)
        print(res.shape)

    def _dinov2_test(self) -> None:
        left_img = torch.rand(1, 3, 448, 448).cuda()
        dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').cuda()
        print(dinov2_vitg14)
        res = dinov2_vitg14(left_img)
        print(res.shape)
        print(res)

    def _depth_test(self) -> None:
        left_img = torch.rand(1, 3, 224, 224).cuda().clone()
        model = dinov2_vitl14_dd().cuda()
        with torch.no_grad():
            res = model(left_img, None, return_loss=True, depth_gt= None)
            print(res.shape)

    def exec(self, args: object) -> None:
        self._depth_test()


def main() -> None:
    unit_test = UnitTest()
    unit_test.exec(None)


if __name__ == '__main__':
    main()
