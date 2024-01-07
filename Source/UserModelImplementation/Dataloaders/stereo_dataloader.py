# -*- coding: utf-8 -*-
import time
import JackFramework as jf
import DatasetHandler as dh
# import UserModelImplementation.user_define as user_def


class StereoDataloader(jf.UserTemplate.DataHandlerTemplate):
    """docstring for DataHandlerTemplate"""
    MODEL_ID = 0                                       # Model
    ID_INTERVAL_STEREO = 2                             # stereo

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args
        self.__result_str, self.__start_time = jf.ResultStr(), 0
        self.__train_dataset, self.__val_dataset = None, None
        self.__saver = dh.StereoSaver(args)

    def get_train_dataset(self, path: str, is_training: bool = True) -> object:
        args = self.__args
        assert args.trainListPath == path
        self.__train_dataset = dh.StereoDataset(args, args.trainListPath, is_training)
        return self.__train_dataset

    def get_val_dataset(self, path: str) -> object:
        args = self.__args
        assert args.valListPath == path
        self.__val_dataset = dh.StereoDataset(args, args.valListPath, False)
        return self.__val_dataset

    def split_data(self, batch_data: tuple, is_training: bool) -> list:
        self.__start_time = time.time()
        if is_training:
            return batch_data[:self.ID_INTERVAL_STEREO], [batch_data[self.ID_INTERVAL_STEREO]]
        return batch_data[:self.ID_INTERVAL_STEREO], batch_data[self.ID_INTERVAL_STEREO:]

    def show_train_result(self, epoch: int, loss: list, acc: list,
                          duration: float) -> None:
        assert len(loss) == len(acc)  # same model number
        jf.log.info(self.__result_str.training_result_str(
            epoch, loss[self.MODEL_ID], acc[self.MODEL_ID], duration, True))

    def show_val_result(self, epoch: int, loss: list, acc: list,
                        duration: float) -> None:
        assert len(loss) == len(acc)  # same model number
        jf.log.info(self.__result_str.training_result_str(
            epoch, loss[self.MODEL_ID], acc[self.MODEL_ID], duration, False))

    def save_result(self, output_data: list, supplement: list,
                    img_id: int, model_id: int) -> None:
        assert self.__train_dataset is not None
        args, off_set = self.__args, 1
        last_position = len(output_data) - off_set
        if model_id == self.MODEL_ID:
            self.__saver.save_output(
                output_data[last_position].cpu().detach().numpy(), img_id,
                args.dataset, supplement, time.time() - self.__start_time)

    def show_intermediate_result(self, epoch: int,
                                 loss: list, acc: list) -> str:
        assert len(loss) == len(acc)  # same model number
        return self.__result_str.training_intermediate_result(
            epoch, loss[self.MODEL_ID], acc[self.MODEL_ID])
