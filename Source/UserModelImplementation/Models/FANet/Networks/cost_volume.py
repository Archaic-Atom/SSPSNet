# -*- coding: utf-8 -*-
import torch


def groupwise_correlation(left_feat: torch.Tensor, right_feat: torch.Tensor,
                          num_groups: int) -> torch.Tensor:
    B, C, H, W = left_feat.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (left_feat * right_feat).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(left_feat: torch.Tensor, right_feat: torch.Tensor,
                     start_disp: int, max_disp: int, num_groups: int):
    B, _, H, W = left_feat.shape
    volume = left_feat.new_zeros([B, num_groups, max_disp, H, W])
    for i in range(max_disp):
        d = start_disp + i
        if d > 0:
            volume[:, :, i, :, d:] = groupwise_correlation(left_feat[:, :, :, d:],
                                                           right_feat[:, :, :, :-d],
                                                           num_groups)
        elif d < 0:
            volume[:, :, i, :, :d] = groupwise_correlation(left_feat[:, :, :, :d],
                                                           right_feat[:, :, :, -d:],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(left_feat, right_feat, num_groups)
    volume = volume.contiguous()
    return volume
