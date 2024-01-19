# -*- coding: utf-8 -*-

import torch
import numpy as np
try:
    from .ganet_deep import GANet
except ImportError:
    from ganet_deep import GANet


class UnitTest(object):
    def __init__(self):
        super().__init__()

    def exec(self, args: object) -> None:
        left_img = torch.rand(1, 3, 240, 528).cuda()
        right_img = torch.rand(1, 3, 240, 528).cuda()

        range_list = torch.from_numpy(np.array([[0, 1]]))
        print(range_list.shape)
        model = GANet(192).cuda()
        num_params = sum(param.numel() for param in model.parameters())
        print(num_params)

        res = model(left_img, right_img)

        print(res)
        print(res[1].shape)

        print("finish")
        # print(model)

        return 0
        for name, param in model.named_parameters():
            print(name)
            if "feature_extraction" in name:
                print(name)
                param.requires_grad = False


def main() -> None:
    unit_test = UnitTest()
    unit_test.exec(None)


if __name__ == '__main__':
    main()
