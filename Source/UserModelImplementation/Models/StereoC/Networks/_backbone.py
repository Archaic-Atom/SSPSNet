# -*- coding: utf-8 -*-
import os
import torch
from torch import nn

ROOT_PATH = '/data3/raozhibo/SAStereo/'
DINOv3_PATH = 'Source/Libs/dinov3/'
DINOv2_PATH = 'Source/Libs/dinov2/'
WEIGHTS_PATH = 'Weights/dinov3/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth'
WEIGHTS_DINOV2_PATH = 'Weights/depth_anything/model.pth'


def _build_dinov3(model_name: str, weights_path: str) -> nn.Module:
    return torch.hub.load(os.path.join(ROOT_PATH, DINOv3_PATH), model_name,
                          source = 'local', weights = weights_path)


def _build_dinov2(model_name: str, weights_path: str) -> nn.Module:
    return torch.hub.load(os.path.join(ROOT_PATH, DINOv2_PATH), model_name,
                          source='local', weights=weights_path)


def _get_backbone_dict() -> dict:
    return {'dinov2': _build_dinov2,
            'dinov3': _build_dinov3}


def get_dino_layers_id(model_name: str) -> list:
    layers_id_dict = {'dinov3_vith16plus': [7, 15, 23, 31],
                      'dinov2_vitl14': [4, 11, 17, 23],
                      }
    assert model_name in layers_id_dict
    return layers_id_dict[model_name]


def create_backbone(backbone_name: str, model_name: str = None,
                    weights_path: str = None) -> nn.Module:
    backnone_dict = _get_backbone_dict()
    assert backbone_name in backnone_dict
    return backnone_dict[backbone_name](model_name, weights_path)


if __name__ == '__main__':
    backbone = create_backbone('dinov3', 'dinov3_vith16plus', os.path.join(ROOT_PATH, WEIGHTS_PATH))
    backbone = backbone.cuda()
    rand_tensor = torch.rand(1, 3, 224, 224).cuda()
    print(backbone)
    output_features = backbone.get_intermediate_layers(
        rand_tensor, n = get_dino_layers_id('dinov3_vith16plus'), reshape = True,
        return_class_token = False, norm = False)
    print(f"Shape of the upsampled output features: {output_features[0].shape}")

    # backbone = create_backbone('dinov2', 'dinov2_vitl14', os.path.join(ROOT_PATH, WEIGHTS_DINOV2_PATH))
    # output_features = backbone.get_intermediate_layers(
    #    rand_tensor, n = [4, 11, 17, 23], reshape = True, return_class_token = False, norm = False)
    # print(f"Shape of the upsampled output features: {output_features[0].shape}")
