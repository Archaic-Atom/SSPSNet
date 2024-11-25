# -*- coding: utf-8 -*-
import torch
from torch import nn
import cv2
from PIL import Image
import numpy as np


class Warp(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _gen_left_coordinate(left_feat: torch.Tensor, device: object) -> torch.Tensor:
        left_coordinate = torch.arange(
            0.0, left_feat.size()[3], device=device).repeat(left_feat.size()[2])
        left_coordinate = left_coordinate.view(left_feat.size()[2], left_feat.size()[3])
        left_coordinate = torch.clamp(left_coordinate, min=0, max=left_feat.size()[3] - 1)
        return left_coordinate.expand(left_feat.size()[0], -1, -1)

    @staticmethod
    def _gen_right_coordinate(left_coordinate: torch.Tensor,
                              disparity_samples: torch.Tensor) -> torch.Tensor:
        return left_coordinate.expand(
            disparity_samples.size()[1], -1, -1, -1).permute([1, 0, 2, 3]) - disparity_samples.float()

    @staticmethod
    def _expand_feat(left_feat: torch.Tensor, right_feat: torch.Tensor,
                     disparity_samples: torch.Tensor) -> tuple:
        # b, c, h, w -> b, c, d, h, w
        right_feature_map = right_feat.expand(
            disparity_samples.size()[1], -1, -1, -1, -1).permute([1, 2, 0, 3, 4])
        left_feature_map = left_feat.expand(
            disparity_samples.size()[1], -1, -1, -1, -1).permute([1, 2, 0, 3, 4])
        return left_feature_map, right_feature_map

    @staticmethod
    def _warp(right_feat: torch.Tensor, right_coordinate: torch.Tensor) -> torch.Tensor:
        right_coordinate = torch.clamp(right_coordinate, min=0, max=right_feat.size()[4] - 1)
        return torch.gather(
            right_feat, dim=4,
            index=right_coordinate.expand(
                right_feat.size()[1], -1, -1, -1, -1).permute([1, 0, 2, 3, 4]).long())

    @staticmethod
    def _occ_warp(warped_right_feature_map: torch.Tensor,
                  right_coordinate: torch.Tensor) -> torch.Tensor:
        return (1 - (
            (right_coordinate.unsqueeze(1) < 0) +
            (right_coordinate.unsqueeze(1) > warped_right_feature_map.size()[4] - 1)
        ).float()) * (warped_right_feature_map) + torch.zeros_like(warped_right_feature_map)

    @staticmethod
    def _squeeze_d(feat: torch.Tensor, id_dim_d: int = 2) -> torch.Tensor:
        # b, c, d, h, w ->  b, c, h, w
        return feat.squeeze(id_dim_d)

    def forward(self, left_feat: torch.Tensor, right_feat: torch.Tensor,
                disparity_samples: torch.Tensor) -> torch.Tensor:
        right_coordinate = self._gen_right_coordinate(
            self._gen_left_coordinate(left_feat, left_feat.get_device()), disparity_samples)

        _, right_feature_map = self._expand_feat(
            left_feat, right_feat, disparity_samples)

        return self._squeeze_d(self._occ_warp(
            self._warp(right_feature_map, right_coordinate), right_coordinate))


def _main() -> None:
    left_img, right_img = cv2.imread('./Example/left.png'), cv2.imread('./Example/right.png')
    disp = np.array(Image.open('./Example/disp.png'), dtype=np.float32) / float(256)
    left_img = torch.tensor(left_img).permute(2, 0, 1).unsqueeze(0).cuda()
    right_img = torch.tensor(right_img).permute(2, 0, 1).unsqueeze(0).cuda()
    disp = torch.tensor(disp).unsqueeze(0).unsqueeze(0).cuda()
    print(left_img.shape, right_img.shape, disp.shape)

    warp = Warp()
    warped_right_feature_map = warp(left_img, right_img, disp)
    print(warped_right_feature_map.shape)

    warped_right_feature_map = warped_right_feature_map.squeeze(0).permute(
        1, 2, 0).cpu().detach().numpy()
    print(warped_right_feature_map.shape)
    cv2.imwrite('./Example/warp_right.png', warped_right_feature_map)

    warped_right_feature_map = warp(torch.ones_like(disp), torch.ones_like(disp), disp)
    warped_right_feature_map = warped_right_feature_map.squeeze(0).permute(
        1, 2, 0).cpu().detach().numpy().astype(np.uint8) * 255
    print(warped_right_feature_map.shape)
    cv2.imwrite('./Example/mask.png', warped_right_feature_map)


if __name__ == '__main__':
    _main()
