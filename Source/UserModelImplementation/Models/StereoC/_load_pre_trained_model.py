# -*- coding: utf-8 -*-
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch import distributed as dist


class LoadPreTrainedModel(object):
    @staticmethod
    def is_module_wrapper(module):
        module_wrappers = (DataParallel, DistributedDataParallel)
        return isinstance(module, module_wrappers)

    @staticmethod
    def load_state_dict(module, state_dict, strict=False, logger=None):
        """Load state_dict to a module.

        This method is modified from :meth:`torch.nn.Module.load_state_dict`.
        Default value for ``strict`` is set to ``False`` and the message for
        param mismatch will be shown even if strict is False.

        Args:
            module (Module): Module that receives the state_dict.
            state_dict (OrderedDict): Weights.
            strict (bool): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
            logger (:obj:`logging.Logger`, optional): Logger to log the error
                message. If not specified, print function will be used.
        """
        unexpected_keys, all_missing_keys, err_msg = [], [], []

        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        # use _load_from_state_dict to enable checkpoint version control
        def load(module, prefix=''):
            # recursively check parallel module in case that the model has a
            # complicated structure, e.g., nn.Module(nn.Module(DDP))
            if LoadPreTrainedModel.is_module_wrapper(module):
                module = module.module
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                         all_missing_keys, unexpected_keys,
                                         err_msg)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(module)
        load = None  # break load->load reference cycle

        # ignore "num_batches_tracked" of BN layers
        missing_keys = [
            key for key in all_missing_keys if 'num_batches_tracked' not in key
        ]

        if unexpected_keys:
            err_msg.append('unexpected key in source '
                           f'state_dict: {", ".join(unexpected_keys)}\n')
        if missing_keys:
            err_msg.append(
                f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

        rank, _ = LoadPreTrainedModel.get_dist_info()
        if len(err_msg) > 0 and rank == 0:
            err_msg.insert(
                0, 'The model and loaded state dict do not match exactly\n')
            err_msg = '\n'.join(err_msg)
            if strict:
                raise RuntimeError(err_msg)
            elif logger is not None:
                logger.warning(err_msg)
            else:
                print(err_msg)

    @staticmethod
    def get_dist_info():
        TORCH_VERSION = torch.__version__
        if TORCH_VERSION < '1.0':
            initialized = dist._initialized
        else:
            if dist.is_available():
                initialized = dist.is_initialized()
            else:
                initialized = False
        if initialized:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        return rank, world_size
