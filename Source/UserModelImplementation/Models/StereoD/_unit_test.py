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
except ImportError:
    from _warp import Warp
    from Networks import Feature


class UnitTest(object):
    def __init__(self):
        super().__init__()
        self._warp = Warp()

    @staticmethod
    def _get_data() -> tuple:
        left_img, right_img = cv2.imread('./Example/left.png'), cv2.imread('./Example/right.png')
        disp = np.array(Image.open('./Example/disp.png'), dtype=np.float32) / float(256)
        left_img, right_img, top_pad, left_pad = UnitTest._padding(left_img, right_img, 384, 1584)
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
        mdoel = self._get_model()
        print(left_img.shape, right_img.shape, disp.shape)

        left_img = F.interpolate(
            mdoel(left_img), [384, 1584], mode='bilinear', align_corners=False)

        right_img = F.interpolate(
            mdoel(right_img), [384, 1584], mode='bilinear', align_corners=False)
        print(left_img.shape, right_img.shape, disp.shape)

        left_img = self._crop_test_img(left_img, top_pad, left_pad)
        right_img = self._crop_test_img(right_img, top_pad, left_pad)
        print(left_img.shape, right_img.shape, disp.shape)

        warped_right_img = self._warp(left_img, right_img, disp)
        mask_occ = self._warp(torch.ones_like(disp), torch.ones_like(disp), disp)
        mask_disp = (disp > 0) & (disp < 192)
        print(type(mask_occ), mask_occ.device)
        print(type(mask_disp), mask_disp.device)
        print(mask_disp)
        print(mask_occ)
        print(mask_disp.bool().float())

        mask = mask_occ * mask_disp.float()
        print(warped_right_img.shape, mask.shape)
        loss = torch.mean(torch.sum(torch.abs(left_img - warped_right_img), dim=1, keepdim=True) * mask)
        print(loss, loss.shape)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        output = torch.mean(cos(left_img, warped_right_img) * mask)
        print(output, output.shape, mask.shape)


def main() -> None:
    unit_test = UnitTest()
    unit_test.exec(None)


if __name__ == '__main__':
    main()
