# -*- coding: utf-8 -*-
import JackFramework as jf
from .stereo_dataloader import StereoDataloader


def _get_dataloaders_dict() -> dict:
    return {'sceneflow': StereoDataloader, 'kitti2012': StereoDataloader,
            'kitti2015': StereoDataloader, 'crestereo': StereoDataloader,
            'eth3d': StereoDataloader, 'rob': StereoDataloader,
            'middlebury': StereoDataloader, 'US3D': StereoDataloader,
            'whu': StereoDataloader,'synthetic': StereoDataloader}


def dataloaders_zoo(args: object, dataset_name: str) -> object:
    dataloader_dict = _get_dataloaders_dict()
    assert dataset_name in dataloader_dict
    return dataloader_dict[dataset_name](args)
