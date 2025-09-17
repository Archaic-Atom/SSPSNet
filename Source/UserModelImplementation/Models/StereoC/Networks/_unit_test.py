# -*- coding: utf-8 -*-
import os
import sys
import torch
from torch import nn
from torchsummary import summary
from einops import rearrange


class UnitTest(object):
    def __init__(self):
        super().__init__()

    def _depth_test(self) -> None:
        g = torch.rand(2, 64, 448, 224).cuda()
        left_img = torch.rand(3, 3, 448, 224).cuda()
        right_img = torch.rand(3, 3, 448, 224).cuda()

        model = StereoA(3, 0, 196, 'dinov2', False).cuda()

    def exec(self, args: object) -> None:
        self._depth_test()


def main() -> None:
    unit_test = UnitTest()
    unit_test.exec(None)


if __name__ == '__main__':
    main()
