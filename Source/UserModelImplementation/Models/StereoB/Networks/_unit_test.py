# -*- coding: utf-8 -*-
import os
import sys
import torch
from torch import nn
from torchsummary import summary
from einops import rearrange

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
    from .model import StereoB
    from ._offset import Refinement, RefinementLayer, MLP
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
    from model import StereoB
    from _offset import Refinement, RefinementLayer, MLP


class UnitTest(object):
    def __init__(self):
        super().__init__()

    def _depth_test(self) -> None:
        left_img = torch.rand(3, 3, 448, 224).cuda()
        right_img = torch.rand(3, 3, 448, 224).cuda()

        model = StereoB(3, 0, 196, 'dinov2', False, 0.1).cuda()

        with torch.no_grad():
            res = model(left_img, right_img)
            print(res)

    def exec(self, args: object) -> None:
        self._depth_test()


def main() -> None:
    unit_test = UnitTest()
    unit_test.exec(None)


if __name__ == '__main__':
    main()
