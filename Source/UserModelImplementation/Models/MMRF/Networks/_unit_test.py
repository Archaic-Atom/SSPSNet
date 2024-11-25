# -*- coding: utf-8 -*-
import os
import sys
import torch
from torchsummary import summary

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
    from .model import FANet
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
    from model import FANet


class UnitTest(object):
    def __init__(self):
        super().__init__()

    def _depth_test(self) -> None:
        left_img = torch.rand(1, 3, 448, 224).cuda()
        model = FANet(3, 0, 196, 'dinov2', False).cuda()
        with torch.no_grad():
            res = model(left_img, left_img)

    def exec(self, args: object) -> None:
        self._depth_test()


def main() -> None:
    unit_test = UnitTest()
    unit_test.exec(None)


if __name__ == '__main__':
    main()
