# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
# This source code is derived from deit but contains significant modifications.
# (https://github.com/facebookresearch/deit)
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the Apache-2.0 license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist
from timm.utils.cuda import NativeScaler
from timm.utils import ModelEma
import logging
from collections import OrderedDict
import numpy as np


_logger = logging.getLogger(__name__)

#next three are future imports from timm 0.4.2
def unitwise_norm(x, norm_type=2.0):
    if x.ndim <= 1:
        return x.norm(norm_type)
    else:
        # works for nn.ConvNd and nn,Linear where output dim is first in the kernel/weight tensor
        # might need special cases for other weights (possibly MHA) where this may not be true
        return x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        if self.count > 0:
            return self.total / self.count
        else:
            return None
    

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t", tb_writer=None, split='train', epoch=0):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.tb_writer = tb_writer
        self.split = split
        self.epoch = epoch

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            if not '_grad_' in name:
                loss_str.append(
                    "{}: {}".format(name, str(meter))
                )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logging.info(print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB)))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
                if not self.split == 'val':
                    for key in self.meters.keys() :
                        if not self.tb_writer is None:
                            iteration = i + self.epoch * len(iterable) if not (self.epoch is None) else i
                            self.tb_writer.add_scalar(self.split+'/'+str(key), self.meters[key].value, iteration )

            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / np.max([1,len(iterable)])))

    def log_val(self, epoch):
        for key in self.meters.keys():
            if not self.tb_writer is None:
                self.tb_writer.add_scalar(self.split + '/' + str(key), self.meters[key].global_avg, epoch)

    def print_val(self, epoch):
        print(f'epoch {epoch}: ' + self.split + f' acc1: {self.meters["acc1"].global_avg}')
        if 'target_digit2' in self.split or 'target_digit3' in self.split:
            print(f'epoch {epoch}: ' + self.split + f' acc2: {self.meters["acc2"].global_avg}')

def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save({'state_dict_ema':checkpoint}, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
# This class is adapted from pytorch-image-models/timm
# (https://github.com/huggingface/pytorch-image-model)
# Hacked together by / Copyright 2021 Ross Wightman
# This source code is licensed under the Apache2.0 license found in the
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
class Retain_Native_Scaler(NativeScaler):
    def __init__(self):
        super().__init__()

    def __call__(self, loss, optimizer, create_graph=False,
                 retain_graph=False, model=None, scale_qkv=False, scale_qkv_mode='mean_grad'):
        self._scaler.scale(loss).backward(create_graph=create_graph, retain_graph=retain_graph)
        if scale_qkv:
            try:
                blocks = model.blocks
                emb_dim = model.embed_dim
            except:
                blocks = model.module.blocks
                emb_dim = model.module.embed_dim
            if scale_qkv_mode == 'mean_grad':
                for blockNr, block in enumerate(blocks):
                    grad = torch.abs(block.attn.qkv.weight.grad)
                    origrad = block.attn.qkv.weight.grad.reshape(3, block.attn.num_heads,
                                                                 emb_dim // block.attn.num_heads, emb_dim)
                    mean_grad = torch.mean(grad)
                    grad = grad.reshape(3, block.attn.num_heads, emb_dim // block.attn.num_heads, emb_dim)
                    for feat in range(grad.shape[0]):
                        feat_factor = mean_grad / torch.mean(grad[feat])
                        origrad[feat] = origrad[feat] * feat_factor
            elif scale_qkv_mode == 'per_head_to_value_grad':
                for blockNr, block in enumerate(blocks):
                    grad = torch.abs(block.attn.qkv.weight.grad)
                    origrad = block.attn.qkv.weight.grad.reshape(3, block.attn.num_heads,
                                                                 emb_dim // block.attn.num_heads, emb_dim)
                    grad = grad.reshape(3, block.attn.num_heads, emb_dim // block.attn.num_heads, emb_dim)
                    for head in range(grad.shape[1]):
                        mean_grad_value = torch.mean(grad[-1, head, :, :])
                        for feat in range(grad.shape[0]):
                            feat_factor = mean_grad_value / torch.mean(grad[feat, head, :, :])
                            origrad[feat, head] = origrad[feat, head] * feat_factor
            elif scale_qkv_mode == 'mean_to_value_grad':
                for blockNr, block in enumerate(blocks):
                    grad = torch.abs(block.attn.qkv.weight.grad)
                    origrad = block.attn.qkv.weight.grad.reshape(3, block.attn.num_heads,
                                                                 emb_dim // block.attn.num_heads, emb_dim)
                    grad = grad.reshape(3, block.attn.num_heads, emb_dim // block.attn.num_heads, emb_dim)
                    mean_grad_value = torch.mean(grad[-1, :, :, :])
                    for feat in range(grad.shape[0]):
                        feat_factor = mean_grad_value / torch.mean(grad[feat, :, :, :])
                        origrad[feat, :] = origrad[feat, :] * feat_factor
            elif scale_qkv_mode == 'match_mean_and_std_per_head':
                for blockNr, block in enumerate(blocks):
                    grad = torch.abs(block.attn.qkv.weight.grad)
                    origrad = block.attn.qkv.weight.grad.reshape(3, block.attn.num_heads,
                                                                 emb_dim // block.attn.num_heads, emb_dim)
                    grad = grad.reshape(3, block.attn.num_heads, emb_dim // block.attn.num_heads, emb_dim)
                    for head in range(grad.shape[1]):
                        mean_grad_value = torch.mean(grad[-1, head, :, :])
                        std_grad_value = torch.std(grad[-1, head, :, :])
                        for feat in range(grad.shape[0]-1):
                            feat_factor = mean_grad_value / torch.mean(grad[feat, head, :, :])
                            # origrad[feat, head] = (origrad[feat, head] * std_grad_value) +

        self._scaler.step(optimizer)
        self._scaler.update()

# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
# This class is adapted from pytorch-image-models/timm
# (https://github.com/huggingface/pytorch-image-model)
# Hacked together by / Copyright 2021 Ross Wightman
# This source code is licensed under the Apache2.0 license found in the
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
class Non_Strict_Model_Ema(ModelEma):
    def __init__(self, model, decay=0.9999, device='', resume=''):
        super().__init__(model, decay=decay, device=device, resume=resume)

    def _load_checkpoint(self, checkpoint_path, strict_reg_mlp=True):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert isinstance(checkpoint, dict)
        if 'state_dict_ema' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict_ema'].items():
                # ema model may have been wrapped by DataParallel, and need module prefix
                if self.ema_has_module:
                    name = 'module.' + k if not k.startswith('module') else k
                else:
                    name = k
                new_state_dict[name] = v
            msg = self.ema.load_state_dict(new_state_dict, strict=not(strict_reg_mlp))
            _logger.info("Loaded state_dict_ema")
            _logger.warning(msg)
        else:
            _logger.warning("Failed to find state_dict_ema, starting from loaded model weights")


def denormalize_img(img):
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    min_val = np.min(img)
    max_val = np.max(img)
    img = (img - min_val)
    img = img / (max_val - min_val)
    return img



