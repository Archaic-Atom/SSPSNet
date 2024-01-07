# -*- coding: utf-8 -*-
# import torch.nn as nn
# import torch.nn.functional as F

import math
import torch
import torch.optim as optim

import JackFramework as jf
import UserModelImplementation.user_define as user_def


class SAStereoInterface(jf.UserTemplate.ModelHandlerTemplate):
    """docstring for DeepLabV3Plus"""
    ID_MODEL = 0
    ID_LEFT_IMG, ID_RIGHT_IMG, ID_DISP_IMG = 0, 1, 2

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args

    @staticmethod
    def lr_lambda(epoch: int) -> float:
        warmup_epochs = 40
        cos_epoch = 1000
        return (epoch / warmup_epochs if epoch < warmup_epochs
                else 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / cos_epoch)))

    def get_model(self) -> list:
        # args = self.__args
        # return model
        return []

    def optimizer(self, model: list, lr: float) -> list:
        args = self.__args
        opt = torch.optim.AdamW(
            model[self.ID_MODEL].parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)

        if args.lr_scheduler:
            sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda=self.lr_lambda)
        else:
            sch = None
        return [opt], [sch]

    def lr_scheduler(self, sch: object, ave_loss: list, sch_id: int) -> None:
        # how to do schenduler
        if self.ID_MODEL == sch_id:
            sch.step()

    def inference(self, model: list, input_data: list, model_id: int) -> list:
        # args = self.__args
        # return output

        return []

    def accuracy(self, output_data: list, label_data: list, model_id: int) -> list:
        # return acc
        # args = self.__args
        args, res, id_three_px = self.__args, [], 1

        if self.MODEL_ID == model_id:
            gt_left = label_data[0]
            mask = (gt_left < args.start_disp + args.disp_num) & (gt_left > args.start_disp)
            for _, item in enumerate(output_data):
                disp = item[:, 0, :, :]
                if len(disp.shape) == 3:
                    acc, mae = jf.acc.SMAccuracy.d_1(disp, gt_left * mask, invalid_value=0)
                    res.extend((acc[id_three_px], mae))
        return res

    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        # return loss
        # args = self.__args
        return []

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
        return False

    # Optional
    def load_opt(self, opt: object, checkpoint: dict, model_id: int) -> bool:
        # return False
        return False

    # Optional
    def save_model(self, epoch: int, model_list: list, opt_list: list) -> dict:
        # return None
        return None
