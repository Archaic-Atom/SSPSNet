# -*- coding: utf-8 -*-
# import torch.nn as nn
import math
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from functools import partial

import JackFramework as jf


import UserModelImplementation.user_define as user_def

from .Networks import FANet
from ._loss import Loss
from ._accuracy import Accuracy


class FANetInterface(jf.UserTemplate.ModelHandlerTemplate):
    """docstring for DeepLabV3Plus"""
    ID_MODEL = 0
    ID_LEFT_DISP_GT = 0
    ID_LEFT_IMG, ID_RIGHT_IMG, ID_DISP_IMG = 0, 1, 2

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args
        self._acc = Accuracy(args)
        self._loss = Loss(args)

    @staticmethod
    def lr_lambda(epoch: int) -> float:
        warmup_epochs = 40
        cos_epoch = 1000
        return (epoch / warmup_epochs if epoch < warmup_epochs
                else 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / cos_epoch)))

    def get_model(self) -> list:
        args = self.__args
        # return model
        model = FANet(3, args.start_disp, args.disp_num, 'dinov2', args.pre_train_opt)

        if not args.pre_train_opt:
            for name, param in model.named_parameters():
                if "feature_extraction" in name:
                    param.requires_grad = False
        return [model]

    def optimizer(self, model: list, lr: float) -> list:
        args = self.__args
        opt = optim.Adam(model[self.ID_MODEL].parameters(), lr=lr, betas=(0.9, 0.999))

        if args.lr_scheduler:
            sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda=self.lr_lambda)
        else:
            sch = None
        return [opt], [sch]

    def lr_scheduler(self, sch: object, ave_loss: list, sch_id: int) -> None:
        # how to do schenduler
        if self.ID_MODEL == sch_id:
            sch.step()

    def inference(self, model: nn.Module, input_data: list, model_id: int) -> list:
        # args = self.__args
        # return output
        if self.ID_MODEL == model_id:
            outputs = jf.Tools.convert2list(model(input_data[self.ID_LEFT_IMG],
                                                  input_data[self.ID_RIGHT_IMG]))
        return outputs

    def accuracy(self, output_data: list, label_data: list, model_id: int) -> list:
        args, acc, id_three_px = self.__args, None, 1

        if self.ID_MODEL == model_id:
            left_img_disp = label_data[self.ID_LEFT_DISP_GT]
            mask = self._get_mask(left_img_disp)

            if args.pre_train_opt:
                acc = self._acc.feature_alignment_accuracy(
                    output_data[self.ID_LEFT_IMG],
                    output_data[self.ID_RIGHT_IMG],
                    left_img_disp, mask)
            else:
                acc = self._acc.matching_accuracy(
                    output_data, left_img_disp * mask, id_three_px)
        return acc

    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        # return loss
        args, loss, id_three_px = self.__args, None, 1
        if self.ID_MODEL == model_id:
            left_img_disp = label_data[self.ID_LEFT_DISP_GT]
            mask = self._get_mask(left_img_disp)

            if args.pre_train_opt:
                loss = self._loss.feature_alignment_loss(
                    output_data[self.ID_LEFT_IMG],
                    output_data[self.ID_RIGHT_IMG],
                    left_img_disp, mask)
            else:
                loss = self._loss.matching_loss(
                    output_data, left_img_disp, mask)
        return loss

    def _get_mask(self, left_img_disp: torch.Tensor) -> torch.Tensor:
        args = self.__args
        return (left_img_disp < args.start_disp + args.disp_num) & (left_img_disp > args.start_disp)

    # Optional
    def pretreatment(self, epoch: int, rank: object) -> None:
        # do something before training epoch
        pass

    # Optional
    def postprocess(self, epoch: int, rank: object,
                    ave_tower_loss: list, ave_tower_acc: list) -> None:
        # do something after training epoch
        pass

    # Optional
    def load_model(self, model: object, checkpoint: dict, model_id: int) -> bool:
        # return False
        # model.load_state_dict(checkpoint['model_0'], strict=False)
        # jf.log.info("Model loaded successfully_add")
        return False

    # Optional
    def load_opt(self, opt: object, checkpoint: dict, model_id: int) -> bool:
        # return False
        return True

    # Optional
    def save_model(self, epoch: int, model_list: list, opt_list: list) -> dict:
        # return None
        return None
