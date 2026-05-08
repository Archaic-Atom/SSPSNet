# -*- coding: utf-8 -*
import JackFramework as jf
from .StereoA import StereoAInterface
from .StereoB import StereoBInterface
from .StereoC import StereoCInterface
from .StereoD import StereoDInterface
from .StereoE import StereoEInterface


def _get_model_dict() -> dict:
    # return {'SAStereo': SAStereoInterface,
    #        'FANet': FANetInterface,
    #        'MMRF': MMRFInterface,
    #       'StereoT': StereoTInterface}
    return {'StereoB': StereoBInterface,
            'StereoA': StereoAInterface,
            'StereoC': StereoCInterface,
            'StereoD': StereoDInterface,
            'StereoE': StereoEInterface}


def model_zoo(args: object, model_name: str) -> object:
    model_dict = _get_model_dict()
    assert model_name in model_dict
    return model_dict[model_name](args)
