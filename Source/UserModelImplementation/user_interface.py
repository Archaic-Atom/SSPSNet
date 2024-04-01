# -*- coding: utf-8 -*-
import argparse
import JackFramework as jf
import UserModelImplementation.user_define as user_def

# model and dataloader
from UserModelImplementation import Models
from UserModelImplementation import Dataloaders


class UserInterface(jf.UserTemplate.NetWorkInferenceTemplate):
    """docstring for UserInterface"""

    def __init__(self) -> object:
        super().__init__()

    def inference(self, args: object) -> object:
        dataloader = Dataloaders.dataloaders_zoo(args, args.dataset)
        model = Models.model_zoo(args, args.modelName)
        return model, dataloader

    def user_parser(self, parser: object) -> object:
        parser.add_argument('--start_disp', type=int, default=user_def.START_DISP,
                            help='start disparity')
        parser.add_argument('--disp_num', type=int, default=user_def.DISP_NUM,
                            help='disparity number')
        parser.add_argument('--lr_scheduler', type=UserInterface.__str2bool,
                            default=user_def.LR_SCHEDULER,
                            help='use or not use lr scheduler')
        parser.add_argument('--pre_train_opt', type=UserInterface.__str2bool,
                            default=user_def.PRE_TRAIN_OPT,
                            help='pre-trained option')
        return parser

    @staticmethod
    def __str2bool(arg: str) -> bool:
        if arg.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
