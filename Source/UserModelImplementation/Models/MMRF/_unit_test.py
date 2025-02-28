# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
import cv2
from PIL import Image
import numpy as np


try:
    from ._warp import Warp
    from .Networks import Feature
    from .models.backbone import SwinAdaptor
except ImportError:
    from _warp import Warp
    from Networks import Feature
    from models.backbone import SwinAdaptor


class UnitTest(object):
    def __init__(self):
        super().__init__()
        self._warp = Warp()

    @staticmethod
    def _get_data() -> tuple:
        left_img, right_img = cv2.imread('./Example/left.png'), cv2.imread('./Example/right.png')
        disp = np.array(Image.open('./Example/disp.png'), dtype=np.float32) / float(256)
        left_img, right_img, top_pad, left_pad = UnitTest._padding(left_img, right_img, 384, 1280)
        left_img = torch.tensor(left_img).permute(2, 0, 1).unsqueeze(0).float().cuda()
        right_img = torch.tensor(right_img).permute(2, 0, 1).unsqueeze(0).float().cuda()
        disp = torch.tensor(disp).unsqueeze(0).unsqueeze(0).float().cuda()
        return left_img, right_img, disp, top_pad, left_pad

    @staticmethod
    def _padding(left_img: np.array, right_img: np.array,
                 padding_height: int, padding_width: int) -> tuple:
        top_pad, left_pad = padding_height - left_img.shape[0], padding_width - right_img.shape[1]
        if top_pad > 0 or left_pad > 0:
            # pading
            left_img = np.lib.pad(left_img, ((top_pad, 0), (0, left_pad), (0, 0)),
                                  mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((top_pad, 0), (0, left_pad), (0, 0)),
                                   mode='constant', constant_values=0)
        return left_img, right_img, top_pad, left_pad

    @staticmethod
    def _crop_test_img(img: torch.Tensor, top_pad: int, left_pad: int) -> torch.Tensor:
        if top_pad > 0 and left_pad > 0:
            img = img[:, :, top_pad:, : -left_pad]
        elif top_pad > 0:
            img = img[:, :, top_pad:, :]
        elif left_pad > 0:
            img = img[:, :, :, :-left_pad]
        return img

    @staticmethod
    def _get_model() -> nn.Module:
        return Feature().cuda()

    def exec(self, args: object) -> None:
        left_img, right_img, disp, top_pad, left_pad = self._get_data()
        print(left_img.shape)

        backbone = SwinAdaptor(out_channels=256, drop_path_rate=0).cuda()
        features = backbone(left_img)
        print([feature.shape for feature in features])


def main() -> None:
    unit_test = UnitTest()
    unit_test.exec(None)


if __name__ == '__main__':
    main()
