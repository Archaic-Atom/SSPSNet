# -*- coding: utf-8 -*-
import torch
import JackFramework as jf


class Accuracy(object):
    DISP_DIM_LEN = 3

    def __init__(self, args: object) -> None:
        super().__init__()
        self.__arg = args

    def matching_accuracy(self, disp_list: list, disp_label: torch.Tensor,
                          id_error_px: int = 1, invalid_value: int = 0) -> list:
        res = []
        for _, disp in enumerate(disp_list):
            if len(disp.shape) == self.DISP_DIM_LEN:
                acc, mae = jf.acc.SMAccuracy.d_1(disp, disp_label, invalid_value)
                res.extend((acc[id_error_px], mae))
        return res
