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
                     start_disp: int, disp_num: int, num_groups: int) -> torch.Tensor:
    B, _, H, W = left_feat.shape
    cost = left_feat.new_zeros([B, num_groups, disp_num, H, W])
    for i in range(disp_num):
        d = start_disp + i
        if d > 0:
            cost[:, :, i, :, d:] = groupwise_correlation(left_feat[:, :, :, d:],
                                                         right_feat[:, :, :, :-d],
                                                         num_groups)
        elif d < 0:
            cost[:, :, i, :, :d] = groupwise_correlation(left_feat[:, :, :, :d],
                                                         right_feat[:, :, :, -d:],
                                                         num_groups)
        else:
            cost[:, :, i, :, :] = groupwise_correlation(left_feat, right_feat, num_groups)
    cost = cost.contiguous()
    return cost


def build_cat_volume(left_feat: torch.Tensor, right_feat: torch.Tensor,
                     start_disp: int, disp_num: int) -> torch.Tensor:
    B, C, H, W = left_feat.shape
    cost = left_feat.new_zeros([B, C * 2, disp_num, H, W])
    for i in range(disp_num):
        d = start_disp + i
        if d > 0:
            cost[:, :left_feat.size()[1], i, :, d:] = left_feat[:, :, :, d:]
            cost[:, left_feat.size()[1]:, i, :, d:] = right_feat[:, :, :, :-d]
        elif d < 0:
            cost[:, :left_feat.size()[1], i, :, :d] = left_feat[:, :, :, :d]
            cost[:, left_feat.size()[1]:, i, :, :d] = right_feat[:, :, :, -d:]
        else:
            cost[:, :left_feat.size()[1], i, :, :] = left_feat
            cost[:, left_feat.size()[1]:, i, :, :] = right_feat
    cost = cost.contiguous()
    return cost
