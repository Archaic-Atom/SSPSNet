# -*- coding: utf-8 -*-
import os
import sys
import torch

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
    from .Mamba import vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
    from .Transformer import build_vit_matching_module
    from .model import FANet
    from Libs.dinov2.dinov2.hub.depthers import dinov2_vitl14_dd
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
    from Mamba import vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
    from Transformer import build_vit_matching_module
    from model import FANet
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

    def _dinov3_test(self) -> None:
        print("Hello")
        left_img = torch.rand(2, 3, 448, 224).cuda()
        right_img = torch.rand(2, 3, 448, 224).cuda()
        dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
        print(dinov2_vitg14)
        res = dinov2_vitg14.get_intermediate_layers(left_img,
                                                    n=1,
                                                    reshape=True,
                                                    return_class_token=False,
                                                    norm=False)
        for i in res:
            print(i.shape)
        out = res[0]
        print(out.shape)

        model = build_vit_matching_module(384).cuda()

        res = model(out)
        for i in res:
            print(i.shape)

    def _dinov2_test(self) -> None:
        left_img = torch.rand(2, 3, 448, 224).cuda()
        right_img = torch.rand(2, 3, 448, 224).cuda()
        print("Hello")

        model = FANet(3, 0, 192, 'dinov2', False).cuda()
        for _ in range(20):
            res = model(left_img, right_img)
            print(res.shape)

    def _multi_gpu_test(self) -> None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        left_img = torch.rand(2, 3, 448, 224).to(device)
        right_img = torch.rand(2, 3, 448, 224).to(device)

        # model = FSNet()
        model = FANet(3, 0, 196, 'dinov2', False)

        # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        net = torch.nn.DataParallel(model).to(device)

        for _ in range(20):
            res = net(left_img, right_img)
            # res = net(left_img)
            print(res.shape)

    def _depth_test(self) -> None:
        left_img = torch.rand(1, 3, 448, 224).cuda()
        model = dinov2_vitl14_dd().cuda()
        with torch.no_grad():
            res = model(left_img, None, return_loss=True, depth_gt= None)
            print(res.shape)

    def exec(self, args: object) -> None:
        self._multi_gpu_test()


def main() -> None:
    unit_test = UnitTest()
    unit_test.exec(None)


if __name__ == '__main__':
    main()
