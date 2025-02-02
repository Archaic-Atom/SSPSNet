# -*- coding: utf-8 -*-
import os
import sys
import torch
from torchsummary import summary

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
    from .model import StereoA
    from ._prompt import Prompt
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
    from model import StereoA
    from _prompt import Prompt


class UnitTest(object):
    def __init__(self):
        super().__init__()

    def _depth_test(self) -> None:
        left_img = torch.rand(1, 8, 448, 224).cuda()
        sparse_depth = torch.rand(1, 1, 448, 224).cuda()
        blur_depth = torch.rand(1, 1, 448, 224).cuda()

        model = Prompt(8, 3)

        with torch.no_grad():
            res = model(left_img, blur_depth, sparse_depth)
            print(res.shape)

    def exec(self, args: object) -> None:
        self._depth_test()


def main() -> None:
    unit_test = UnitTest()
    unit_test.exec(None)


if __name__ == '__main__':
    main()
