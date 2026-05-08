# -*- coding: utf-8 -*-
from collections import OrderedDict
import math
import torch
import torch.optim as optim
from torch import nn

import JackFramework as jf

import UserModelImplementation.user_define as user_def
from UserModelImplementation.Models.StereoA._load_pre_trained_model import LoadPreTrainedModel

from .Networks import StereoE
from ._loss import Loss
from ._accuracy import Accuracy


class StereoEInterface(jf.UserTemplate.ModelHandlerTemplate):
    """High-performance sparse prompt stereo interface."""
    ID_MODEL = 0
    ID_LEFT_DISP_GT = 0
    ID_LEFT_IMG, ID_RIGHT_IMG = 0, 1

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args
        self._acc = Accuracy(args)
        self._loss = Loss(args)
        self._sch = None
        self._first_opt = True

    @staticmethod
    def lr_lambda(epoch: int) -> float:
        warmup_epochs = 20
        cos_epoch = 800
        return (epoch / warmup_epochs if epoch < warmup_epochs
                else 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / cos_epoch)))

    def get_model(self) -> list:
        args = self.__args
        backbone = getattr(args, "stereoBackbone", "dinov3")
        backbone_variant = getattr(args, "stereoBackboneVariant", None)
        backbone_weights = getattr(args, "stereoBackboneWeights", None)

        model = StereoE(
            3, args.start_disp, args.disp_num, backbone,
            backbone_variant=backbone_variant,
            backbone_weights=backbone_weights,
            pre_train_opt=args.pre_train_opt,
            confidence_level=args.confidence_level,
            prompt_min_conf=getattr(args, "prompt_min_conf", 0.65)
        )
        return [model]

    def optimizer(self, model: list, lr: float) -> list:
        args = self.__args
        opt = optim.AdamW(model[self.ID_MODEL].parameters(), lr=lr, weight_decay=1e-4)
        if args.lr_scheduler:
            sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda=self.lr_lambda)
        else:
            sch = None
        self._sch = sch
        return [opt], [sch]

    def lr_scheduler(self, sch: object, ave_loss: list, sch_id: int) -> None:
        if self.ID_MODEL == sch_id and sch is not None:
            sch.step()

    def inference(self, model: nn.Module, input_data: list, model_id: int) -> list:
        if self.ID_MODEL == model_id:
            outputs = jf.Tools.convert2list(model(input_data[self.ID_LEFT_IMG],
                                                  input_data[self.ID_RIGHT_IMG]))
            if self._sch is not None:
                if self._first_opt:
                    self._first_opt = False
                else:
                    self._sch.step()
        return outputs

    def accuracy(self, output_data: list, label_data: list, model_id: int) -> list:
        args = self.__args
        if self.ID_MODEL != model_id:
            return None

        left_img_disp = label_data[self.ID_LEFT_DISP_GT]
        mask = self._get_mask(left_img_disp)

        if args.pre_train_opt:
            left_feat, right_feat = output_data
            return self._acc.feature_alignment_accuracy(
                left_feat, right_feat, left_img_disp, mask)
        return self._acc.matching_accuracy(output_data, left_img_disp * mask)

    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        args = self.__args
        if self.ID_MODEL != model_id:
            return None
        left_img_disp = label_data[self.ID_LEFT_DISP_GT]
        mask = self._get_mask(left_img_disp)

        if args.pre_train_opt:
            left_feat, right_feat = output_data
            return self._loss.feature_alignment_loss(
                left_feat, right_feat, left_img_disp, mask)
        return self._loss.matching_loss(output_data, left_img_disp, mask)

    def _get_mask(self, left_img_disp: torch.Tensor) -> torch.Tensor:
        args = self.__args
        return (left_img_disp < args.start_disp + args.disp_num) & (left_img_disp > args.start_disp)

    @staticmethod
    def _load_pre_trained_model(model: object, checkpoint: dict) -> None:
        state_dict, off_set = OrderedDict(), 1
        old_model_name = 'pipeline'
        for key, value in checkpoint['state_dict'].items():
            if old_model_name in key:
                new_key = key[len(old_model_name) + off_set:]
                state_dict[new_key] = value
            else:
                state_dict[key] = value

        load_pre_trained_model = LoadPreTrainedModel()
        load_pre_trained_model.load_state_dict(model, state_dict)

    def load_model(self, model: object, checkpoint: dict, model_id: int) -> bool:
        assert model_id == self.ID_MODEL
        args = self.__args
        if args.load_pre_train_model_opt:
            self._load_pre_trained_model(model, checkpoint)
            jf.log.info("load the pretrained model")
        else:
            model.load_state_dict(checkpoint['model_0'], strict=False)
            jf.log.info("load the old model")
        return True

    def load_opt(self, opt: object, checkpoint: dict, model_id: int) -> bool:
        if getattr(self.__args, "load_pre_train_model_opt", False):
            return True
        opt_name = f'opt_{model_id}'
        if opt_name in checkpoint:
            opt.load_state_dict(checkpoint[opt_name])
        return True

    def save_model(self, epoch: int, model_list: list, opt_list: list) -> dict:
        assert len(model_list) == len(opt_list)
        model_dict = {'epoch': epoch}
        for i, _ in enumerate(model_list):
            model_name = f'model_{i}'
            opt_name = f'opt_{i}'
            model_dict[model_name] = model_list[i].state_dict()
            model_dict[opt_name] = opt_list[i].state_dict()
        return model_dict
