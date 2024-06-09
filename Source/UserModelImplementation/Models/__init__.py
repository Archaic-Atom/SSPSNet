# -*- coding: utf-8 -*
import JackFramework as jf
from .SAStereo import SAStereoInterface
from .FANet import FANetInterface


def _get_model_dict() -> dict:
    return {'SAStereo': SAStereoInterface,
            'FANet': FANetInterface, }


def model_zoo(args: object, model_name: str) -> object:
    model_dict = _get_model_dict()
    assert model_name in model_dict
    return model_dict[model_name](args)
