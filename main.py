# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
# This source code is derived from deit but contains significant modifications.
# (https://github.com/facebookresearch/deit)
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the Apache-2.0 license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
from pathlib import Path

from model_lib import model_lib, load_checkpoint, load_checkpoint_to_resume_training
from timm.loss import LabelSmoothingCrossEntropy
from timm.scheduler import create_scheduler
from helpers import get_temperature_schedule
from optim_factory_own import create_optimizer
from timm.utils import get_state_dict

from datasets import build_dataset
from engine import train_one_epoch, evaluate, train_one_epoch_resnet, \
    train_one_epoch_linear_probes, evaluate_linear_probe_all_heads
from losses import DistillationLoss
from samplers import RASampler
import os
from utils import Retain_Native_Scaler, Non_Strict_Model_Ema
from probe_utils import LinearProbeCollection

import sys
sys.path.append(os.getcwd()+'/..')
import utils
from torch.utils.tensorboard import SummaryWriter

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)

    parser.add_argument('--img_size', default=56, type=int)

    parser.add_argument('--qkv_bias', default=None, type=str2bool)
    parser.add_argument('--pos_embed_type', default='learnable', type=str)

    # basic settings
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='vanilla_vit_mnist_patch7_56', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--dropout_head', type=float, default=0.0, 
                        help='dropout rate on dropout before cls head')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, 
                        help='in attention block')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # gradient scaling baseline
    parser.add_argument('--scale_qkv_grad', default=False, type=str2bool, help='')
    parser.add_argument('--scale_qkv_mode', default='mean_grad', type=str, 
                        help='one of mean_grad, per_head_to_value_grad')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    parser.add_argument('--src', action='store_true')  # simple random crop

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--lin_eval', default='', help='linear evaluation from checkpoint')
    parser.add_argument('--dataset_subset_fraction', default=1.0, type=float, 
                        help='used for linprobe experiments to reduce dataset size for online codelength')
    parser.add_argument('--lr_head', default=None, type=float, help='use a different lr on the cls head')

    # Dataset parameters
    parser.add_argument('--data-set', default='IMNET',
                        choices=[ 'MNIST_spacial_decision'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--always_top_right', default=False, type=str2bool)
    parser.add_argument('--indicator_subset_fraction', default=1., type=float)
    parser.add_argument('--valid-data-set', default=None,
                        choices=['MNIST_spacial_decision'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--save_eval_set', default=False, type=str2bool)
    parser.add_argument('--eval_set_save_path', default='./data', type=str)
    parser.add_argument('--use_saved_eval_set', default=False, type=str2bool)
    parser.add_argument('--end_after_eval', default=False, type=str2bool) 
    parser.add_argument('--eval_step', default=None, type=int) 
    parser.add_argument('--save_intermed_checkpoints', default=False, type=str2bool)
    parser.add_argument('--attention_type', type=str, default='softmax', 
                        choices=['softmax', 'norm_softmax', 'norm_softmax_std'])

    #MNIST options
    parser.add_argument('--colors_per_class', default=1, type=int, help='how many colors are used for each digit')
    parser.add_argument('--mnist_task', type=str, default='spacial_decision_indicator_digit_1_2_fashion', 
                        choices=['spacial_decision_indicator_digit_1_2_fashion',
                                'sd_indicator_digit_1_2_fashion_no_fixed_pos',
                                'mnist_fashion_cifar_ind',
                                'fashion_position_as_indicator_topIfAbove',
                                'same_color_distractors_colored_target',
                                'spacial_decision_indicator_digit_1_2_colored_target_or_fashion',
                                'special_decision_digit_group_4_fashion'])
    parser.add_argument('--embedding_dim_per_head', default=64, type=int)
    parser.add_argument('--mnist_deit_depth', default=7, type=int)
    parser.add_argument('--mlp_ratio', default=2, type=int)
    parser.add_argument('--num_mnist_targets', default=1, type=int,
                        help='how many different targets should be returned (digit or digit+color)')
    parser.add_argument('--save_for_linprobing', default=False, type=str2bool)
    parser.add_argument('--patch_size', default=4, type=int)
    parser.add_argument('--top_right_probability', default=0.5, type=float)
    parser.add_argument('--layer_norm_eps', default=1e-6, type=float)

    # general stuff
    parser.add_argument('--output_dir', default='./outputs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain', default=None, help='pretrained checkpoint used for additional finetuning')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--subset_fraction', type=float, default=1.0)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    #eval and plot settings
    parser.add_argument('--get_class_accuracies', default=False, type=str2bool)
    parser.add_argument('--return_attention', type=str2bool, default=False)
    parser.add_argument('--qkv_grad_plot', type=str2bool, default=False)
    parser.add_argument('--log_attention', type=str2bool, default=False)
    parser.add_argument('--log_n_attention_images', type=int, default=1)
    parser.add_argument('--log_abs_gradient', type=str2bool, default=True)
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--linear_probes_all_heads', type=str2bool, default=False)
    parser.add_argument('--mlp_probes', type=str2bool, default=False)
    parser.add_argument('--linprobe_after_residuals', type=str2bool, default=False)
    parser.add_argument('--cls_token_linprobe', type=str2bool, default=False)

    # more baseline experiments
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--return_z', nargs='+')
    parser.add_argument('--return_qkv', default=False, type=str2bool)
    parser.add_argument('--return_intermed_x', default=False, type=str2bool)
    parser.add_argument('--temperature_annealing', default=False, type=str2bool)
    parser.add_argument('--start_temperature', default=0.125, type=float)
    parser.add_argument('--end_temperature', default=0.125, type=float)
    parser.add_argument('--temperature_schedule', default='linear', type=str, help='linear or cosine')
    parser.add_argument('--decay_rate_temp', default=0.1, type=float)

    args = parser.parse_args()

    # modify args and check for unsupported settings
    if not args.return_z is None:
        if len(args.return_z) == 1:
            args.return_z = args.return_z[0] == "True"
    if args.model == 'vanilla':
        args.model = 'vanilla_vit_mnist_patch7_56'
    if args.dataset_subset_fraction < 1 and args.lin_eval == '':
        raise Exception('dataset_subst_fraction only supported for linear probe')
    if args.world_size>1:
        raise Exception('final code has not been tested for multi-gpu setting, use with care')

    return args


def main(args):
    utils.init_distributed_mode(args)
    print(args)
    os.makedirs(args.output_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=args.output_dir)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # get dataset 
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    if args.dataset_subset_fraction < 1 and not args.lin_eval == '':
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)   
    else:
        sampler_train = RASampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # in case batch size is larger than the number of samples 
    # change batch size and correct the learning rate
    if not args.lin_eval == '':
        drop_last = False
        if args.batch_size > len(sampler_train):
            prev_bs = args.batch_size
            args.batch_size = len(sampler_train)
            linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / prev_bs
            args.lr = linear_scaled_lr
    else:
        drop_last = True

    # get dataloaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=drop_last)
    eval_bs = int(1.5 * args.batch_size)       
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=eval_bs,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False)

    # create model
    print(f"Creating model: {args.model}")
    model = model_lib(args, dataset_train.nb_classes)
    # load model if lin_eval or finetune is set. resume handled below
    model, head_params = load_checkpoint(args, model)
    model.to(device)

    # set up ema
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, 
        # and AMP but before SyncBN and DDP wrapper
        model_ema = Non_Strict_Model_Ema(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    # create ddp
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], 
                                                          find_unused_parameters=False)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # set up optimizers and learning rate
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp, exclude_params=[])
    loss_scaler = Retain_Native_Scaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    # if training should be resumed load checkpoint, optimizer and lr_scheduler
    if args.resume:
        args, model_without_ddp, optimizer, \
            lr_scheduler, model_ema, loss_scaler = \
                load_checkpoint_to_resume_training(
                    args, model_without_ddp, optimizer, lr_scheduler, model_ema, loss_scaler)

    
    if args.linear_probes_all_heads:
        # set up linear probes 
        lin_probes = LinearProbeCollection(model, args.nb_classes, args.num_mnist_targets, args)
        lin_probes.set_trainable()
        lin_probes.to_gpu()
        lp_optimizers = lin_probes.get_optimizers(args, create_optimizer)

    # get loss function
    if args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    teacher_model = None
    criterion = DistillationLoss(
        criterion, teacher_model, 'none', 0, 0)

    # define temperature schedule if used
    temperatures = None
    if args.temperature_annealing:
        temperatures = get_temperature_schedule(args)
    
    if args.lin_eval:        
        # freeze all layers and only train the classification head
        for name_p, p in model.named_parameters():
            if 'resnet' in args.model:
                if any(hp in name_p for hp in head_params):
                    p.requires_grad = True
                    print(f"Train model weights: {name_p}")
                else:
                    p.requires_grad = False
            else:
                if 'head' in name_p:
                    p.requires_grad = True
                    print(f"Train model weights: {name_p}")
                else:
                    p.requires_grad = False

    # run only evaluation
    if args.eval:
        file_ending = '.npy'

        if not (args.get_class_accuracies):
            eval_out = evaluate(data_loader_val, model, device, args, tb_writer)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {eval_out['test_stats']['acc1']:.1f}%")
            return
        elif args.get_class_accuracies:
            eval_out = evaluate(data_loader_val, model, device, args, tb_writer)
            if len(args.finetune)>0:
                model_name = args.finetune.split('/')[-2]
            else:
                model_name = args.resume.split('/')[-2]
            if args.get_class_accuracies:
                class_accs = eval_out['class_accs']
                np.save(os.path.join(args.output_dir, model_name + '_class_acccs' + file_ending), 
                    class_accs)
            return

    # get model params for gradient clipping during training
    model_params = []
    for _, p in model_without_ddp.named_parameters():
        model_params.append(p)

    #start training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if 'resnet' in args.model:
            train_stats = train_one_epoch_resnet(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                set_training_mode=args.train_mode,
                tb_writer=tb_writer,
            )
        elif args.linear_probes_all_heads:
            train_stats = train_one_epoch_linear_probes(
                model, criterion, data_loader_train,
                lp_optimizers, device, epoch, loss_scaler,
                args=args, tb_writer=tb_writer,
                linear_probes=lin_probes,
                targets=args.num_mnist_targets,
                temperatures=temperatures,
            )
        else:
            train_stats = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                model_ema,
                set_training_mode=args.train_mode,
                args=args, tb_writer=tb_writer,
                temperatures=temperatures,
            )
        lr_scheduler.step(epoch)

        checkpoint_paths = [os.path.join(args.output_dir, 'checkpoint.pth')]
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'model_ema': get_state_dict(model_ema),
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }, checkpoint_path)


        if args.linear_probes_all_heads:
            evaluate_linear_probe_all_heads(data_loader_val, lin_probes, model, device, args, tb_writer=tb_writer,
                                            epoch=epoch, targets=args.num_mnist_targets, temperatures=temperatures)
            test_stats = None
        else:
            eval_out = evaluate(data_loader_val, model, device, args, tb_writer, epoch, temperatures=temperatures)
            test_stats = eval_out['test_stats']

        if not test_stats is None:
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir:
                    checkpoint_paths = [os.path.join(args.output_dir, 'best_checkpoint.pth')]
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'model_ema': get_state_dict(model_ema),
                            'scaler': loss_scaler.state_dict(),
                            'args': args,
                        }, checkpoint_path)
            print(f'Max accuracy: {max_accuracy:.2f}%')
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            if args.output_dir and utils.is_main_process():
                log_stats = {k:v for k,v in log_stats.items() if not '_grad_' in k}
                with (Path(os.path.join(args.output_dir, "log.txt"))).open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)



