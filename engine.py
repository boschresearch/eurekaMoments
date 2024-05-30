# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#This source code is derived from deit but contains significant modifications.
#  (https://github.com/facebookresearch/deit)
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the Apache-2.0 license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.
"""
Train and eval functions used in main.py
"""
import os
import math
import sys
from typing import Iterable, Optional
import torch
from timm.utils import ModelEma
from losses import DistillationLoss
import utils
import numpy as np
import io
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, 
                    model_ema: Optional[ModelEma] = None,
                    set_training_mode=True, args=None, tb_writer=None,
                    temperatures=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ", tb_writer=tb_writer, epoch=epoch)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for step_nr, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        samples = {'x': samples}
        samples['scale'] = None
        if not temperatures is None:
            samples['scale'] = temperatures[epoch]

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if not temperatures is None:
                samples = samples['x']
            loss = criterion(samples, outputs['x'], targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            raise Exception('Loss is {}'.format(loss_value))

        optimizer.zero_grad()
        if args.lr > 0.0:
            loss_scaler(loss, optimizer, 
                        create_graph=False, model=model,
                        scale_qkv=args.scale_qkv_grad, 
                        scale_qkv_mode=args.scale_qkv_mode)
        
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        # logging
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if args.log_abs_gradient:
            metric_logger = log_abs_grad(metric_logger, model=model)
        if args.qkv_grad_plot and step_nr == 0:
            metric_logger = log_qkv_grad(metric_logger, epoch, outputs)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_resnet(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, 
                    set_training_mode=True, tb_writer=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ", tb_writer=tb_writer, epoch=epoch)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()  
        loss_scaler(loss, optimizer, create_graph=False)
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_linear_probes(
        model: torch.nn.Module, criterion: DistillationLoss, data_loader: Iterable, 
        optimizer: [torch.optim.Optimizer], device: torch.device, epoch: int, 
        loss_scaler, linear_probes=None, targets=1,
        target_list=['digit1', 'digit2', 'digit3', 'digit4', 'color1', 'color2', 
                     'color3', 'color4', 'target_location'], 
        args=None, tb_writer=None, temperatures=None):
    model.eval()
    linear_probes.train()
    linear_probes = linear_probes.probe_list
    num_heads = model.num_heads if not isinstance(
        model, torch.nn.parallel.DistributedDataParallel) else model.module.num_heads
    metric_logger = utils.MetricLogger(delimiter="  ", tb_writer=tb_writer, epoch=epoch)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_loggers = []

    if args.num_mnist_targets == 6:
        target_list = ['digit1', 'digit2', 'digit3', 'digit4', 'target_location', 'true_label']
    if args.return_qkv:
        reps = ['z', 'q', 'k', 'v']
    else:
        reps = ['z']
    if args.return_intermed_x:
        reps.append('x_intermed')

    for d in range(args.mnist_deit_depth):
        ml_heads = []
        for i in range(num_heads):
            ml_targets = []
            for target in range(targets):
                ml_reps = []
                for rep in reps:
                    ml_reps.append(utils.MetricLogger(
                        delimiter="  ", tb_writer=tb_writer, 
                        split=f'lin_probe_train_layer{d}_head{i}_target_{target_list[target]}_feature_{rep}'))
                ml_targets.append(ml_reps)
            ml_heads.append(ml_targets)
        metric_loggers.append(ml_heads)
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for step_nr, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 
                                                                         print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if len(targets.shape) == 1:
            targets = targets.reshape(-1,1)
        samples = {'x': samples}
        samples['scale'] = None
        if not temperatures is None:
            samples['scale'] = temperatures[epoch]

        with torch.cuda.amp.autocast():
            outputs = model(samples)
        for layer in range(len(linear_probes)):
            for head in range(len(linear_probes[layer])):
                for rep_nr, rep in enumerate(reps):
                    if len(outputs[rep].shape) == 4:
                        if rep =='x_intermed':
                            if head > 0:
                                continue
                            outs = outputs[rep].reshape(outs.shape[0], args.mnist_deit_depth, -1)
                        else:
                            outs = outputs[rep].reshape(outs.shape[0], args.mnist_deit_depth, 
                                                        args.num_heads, 65, 64)
                    else:
                        outs = outputs[rep]

                    for target in range(len(linear_probes[layer][head])):
                        with torch.cuda.amp.autocast():
                            if args.cls_token_linprobe:
                                if rep == 'x_intermed':
                                    outs = outputs[rep][:, layer, 0,:]
                                    probe_out = linear_probes[layer][head][target][rep_nr](
                                        outs.detach())
                                else:
                                    probe_out = linear_probes[layer][head][target][rep_nr](
                                        outs[:, layer, head, 0, :].reshape(outs.shape[0], -1).detach())
                            else:
                                if rep == 'x_intermed':
                                    probe_out = linear_probes[layer][head][target][rep_nr](
                                        outs[:, layer, :].reshape(outs.shape[0], -1).detach())
                                else:
                                    probe_out = linear_probes[layer][head][target][rep_nr](
                                        outs[:, layer, head, :].reshape(outs.shape[0], -1).detach())
                                    
                        loss = criterion(samples, probe_out, targets[:, target].squeeze())
                        optimizer[layer][head][target][rep_nr].zero_grad()
                        loss_scaler(loss, optimizer[layer][head][target][rep_nr], create_graph=False)
                        loss_value = loss.item()
                        torch.cuda.synchronize()
                        metric_loggers[layer][head][target][rep_nr].update(loss=loss_value)
        metric_logger.update(lr=optimizer[0][0][0][0].param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def log_abs_grad(metric_logger, model):
    names = ['q', 'k', 'v']
    with torch.no_grad():
        try:
            blocks = model.blocks
            emb_dim = model.embed_dim
        except:
            blocks = model.module.blocks
            emb_dim = model.module.embed_dim
        for blockNr, block in enumerate(blocks):
            grad = torch.abs(block.attn.qkv.weight.grad.detach().clone())
            ins = {f'mean_abs_grad_layer{blockNr}': torch.mean(grad)}
            metric_logger.update(**ins)
            grad = grad.reshape(3, block.attn.num_heads, emb_dim // block.attn.num_heads, emb_dim)
            for qkv in range(3):
                ins = {f'mean_abs_grad_layer{blockNr}_{names[qkv]}': torch.mean(grad[qkv])}
                metric_logger.update(**ins)    
    return metric_logger


def log_qkv_grad(metric_logger, epoch, outputs):
    for feat in ['grad_q', 'grad_k', 'grad_v']:
        gr = torch.abs(outputs[feat])
        for layer_nr in range(gr.shape[1]):
            for head_nr in range(gr.shape[2]):
                map = gr[:, layer_nr, head_nr, :, :].mean(dim=-1).mean(dim=0)[1:].reshape(14, 14)
                grad_indi = torch.cat([map[0:7, 0:7].reshape(-1), map[7:, 7:].reshape(-1)], dim=0)
                grad_target = torch.cat([map[0:7, 7:].reshape(-1), map[7:, 0:7].reshape(-1)], dim=0)

                ins = {f'target_grad_feature_{feat}_layer_{layer_nr}_head_{head_nr}': 
                       torch.mean(grad_target)}
                metric_logger.update(**ins)
                ins = {f'indicator_grad_feature_{feat}_layer_{layer_nr}_head_{head_nr}': 
                       torch.mean(grad_indi)}
                metric_logger.update(**ins)
                map = map.cpu().numpy()
                img_arr = log_grad_qkv(map)
                metric_logger.tb_writer.add_image(
                    f'Grad_imgs_layer_{feat}_{layer_nr}_head_{head_nr}', 
                    img_arr.transpose(2, 0, 1).astype(np.uint8), global_step=epoch)

            # get mean over heads
            map = gr[:, layer_nr, :, :, :].mean(dim=-1).mean(-2).mean(dim=0)[1:].reshape(14, 14)
            grad_indi = torch.cat([map[0:7, 0:7].reshape(-1), map[7:, 7:].reshape(-1)], dim=0)
            grad_target = torch.cat([map[0:7, 7:].reshape(-1), map[7:, 0:7].reshape(-1)], dim=0)

            ins = {f'target_mean_grad_feature_{feat}_layer_{layer_nr}': torch.mean(grad_target)}
            metric_logger.update(**ins)
            ins = {f'indicator_mean_grad_feature_{feat}_layer_{layer_nr}': torch.mean(grad_indi)}
            metric_logger.update(**ins)
            map = map.cpu().numpy()
            img_arr = log_grad_qkv(map)
            metric_logger.tb_writer.add_image(
                f'Grad_imgs_layer_{feat}_{layer_nr}_mean',
                img_arr.transpose(2, 0, 1).astype(np.uint8), global_step=epoch)
            
    return metric_logger


def accuracy(output, target, topk=(1,), get_correct=False, num_classes=np.inf):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    if maxk > num_classes:
        maxk = num_classes

    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    if not get_correct:
        return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]
    else:
        return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk], correct[:1]


def save_checkpoint(args, model_without_ddp, optimizer, lr_scheduler, epoch,step, loss_scaler):
    output_dir = Path(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    checkpoint_paths = [output_dir / f'_epoch{epoch}_step_{step}_checkpoint.pth']
    for checkpoint_path in checkpoint_paths:
        utils.save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
        }, checkpoint_path)
        

@torch.no_grad()
def get_class_accuracies(class_accs, idx_to_classes, output, target):
    for targ in torch.unique(target):
        t_idx = targ == target
        acc1, acc5 = accuracy(output['x'][t_idx], target[t_idx], topk=(1, 5))
        c = idx_to_classes[str(targ.cpu().numpy())][1]
        class_accs[c].meters['acc1 ' + c].update(acc1.item(), n=sum(t_idx))
    return class_accs


def class_accs_2_dict(class_accs):
    class_acc_dict_acc1 = {}
    class_acc_dict_acc5 = {}
    for c in class_accs.keys():
        try:
            class_acc_dict_acc1[c] = class_accs[c].meters['acc1'].global_avg.cpu().numpy()
        except ZeroDivisionError:
            class_acc_dict_acc1[c] = None
            class_acc_dict_acc5[c] = None
        class_acc_dict_acc5[c] = None
    return [class_acc_dict_acc1, class_acc_dict_acc5]


@torch.no_grad()
def evaluate_deit(data_loader, model, device, args, tb_writer=None, epoch=None, temperatures=None):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ", tb_writer=tb_writer, split='val')
    header = 'Test:'

    if args.get_class_accuracies:
        class_accs = {}
        idx_to_classes = {str(v):[k, k] for k,v in data_loader.dataset.class_to_idx.items()}

        for c, v in data_loader.dataset.class_to_idx.items():
            name = idx_to_classes[str(v)][1]
            class_accs[name] = utils.MetricLogger(delimiter="  ", tb_writer=tb_writer, split='val')

    # switch to evaluation mode
    model.eval()
    save_idx = 0
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        ims = images
        target = target.to(device, non_blocking=True)
        images = {'x': images}
        images['scale'] = None
        if not temperatures is None:
            images['scale'] = temperatures[epoch]

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            if not temperatures is None:
                images = ims
            
            loss = criterion(output['x'], target)

        try:
            num_classes = len(data_loader.dataset.classes)
        except:
            num_classes = data_loader.dataset.nb_classes
        
        acc1, acc2, acc5 = accuracy(output['x'], target, topk=(1, 2, 5),
                                    num_classes=num_classes)
        
        if isinstance(images, dict):
            images = images['x']
        if not ims is None:
            images = ims
        batch_size = images.shape[0]
        
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc2'].update(acc2.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        if args.get_class_accuracies:
            class_accs = get_class_accuracies(class_accs, idx_to_classes, output, target)

        #visualize attention maps if attention gets returned
        if args.return_attention:
            if args.log_attention and save_idx == 0:
                has_cls = True if 'vanilla' in args.model or 'deit' in args.model else False
                for img_nr in range(args.log_n_attention_images):
                    img = log_attn(img_nr, output['attn'].cpu(), images, args, target, has_cls)
                    metric_logger.tb_writer.add_image(
                        f'ValImages_{img_nr}', img.transpose(2, 0, 1).astype(np.uint8),
                        global_step=epoch)
            save_idx += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    if not epoch is None:
        metric_logger.log_val(epoch)

    returns = {}
    if args.get_class_accuracies:
        for k in class_accs.keys():
            class_accs[k].synchronize_between_processes()
            class_accs[k].log_val(epoch)
        returns['class_accs'] = class_accs_2_dict(class_accs)

    returns['test_stats'] = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return returns


@torch.no_grad()
def evaluate_resnet(data_loader, model, device, args, tb_writer, epoch):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ", tb_writer=tb_writer, split='val')
    header = 'Test:'
    model.eval()
    if args.get_class_accuracies:
        pass
        class_accs = {}
        idx_to_classes = {}
        for c, v in data_loader.dataset.class_to_idx.items():
            class_accs[c] = utils.MetricLogger(delimiter="  ", split='val')
            idx_to_classes[str(v)] = c

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        try:
            num_classes = len(data_loader.dataset.classes)
        except:
            num_classes = data_loader.dataset.nb_classes

        acc1, acc5 = accuracy(output, target, topk=(1, 5),  
                              num_classes=num_classes)
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        if args.get_class_accuracies:
            class_accs = get_class_accuracies(class_accs, idx_to_classes, {'x':output}, target)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    if not epoch is None:
        metric_logger.log_val(epoch)

    returns = {}
    returns['test_stats'] = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if args.get_class_accuracies:
        returns['class_accs'] = class_accs_2_dict(class_accs)

    return returns


@torch.no_grad()
def evaluate_linear_probe_all_heads(
    data_loader, linear_probes, model, device, args,
    target_list=['digit1', 'digit2', 'digit3', 'digit4', 'color1', 'color2', 'color3',
                 'color4', 'target_location'],
    targets=1, tb_writer=None, epoch=None, temperatures=None):
    criterion = torch.nn.CrossEntropyLoss()
    num_heads = model.num_heads if not isinstance(
        model, torch.nn.parallel.DistributedDataParallel) else model.module.num_heads
    if args.return_qkv:
        reps = ['z', 'q', 'k', 'v']
    else:
        reps = ['z']
    if args.return_intermed_x:
        reps.append('x_intermed')
    if args.num_mnist_targets == 1:
        target_list = ['target_location']
    if args.num_mnist_targets == 6:
        target_list = ['digit1', 'digit2', 'digit3', 'digit4', 'target_location', 'true_label']
    metric_logger = utils.MetricLogger(delimiter="  ", tb_writer=tb_writer, epoch=epoch)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_loggers = []
    for d in range(args.mnist_deit_depth):
        ml_heads = []
        for i in range(num_heads):
            ml_targets = []
            for target in range(targets):
                ml_reps = []
                for rep in reps:
                    ml_reps.append(
                        utils.MetricLogger(
                            delimiter="  ", tb_writer=tb_writer, 
                            split=f'lin_probe_test_layer{d}_head{i}_target_{target_list[target]}_features_{rep}'))
                ml_targets.append(ml_reps)
            ml_heads.append(ml_targets)
        metric_loggers.append(ml_heads)
    linear_probes.eval()
    linear_probes = linear_probes.probe_list

    for step_nr, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if len(targets.shape) == 1:
            targets = targets.reshape(-1, 1)
        samples = {'x': samples}
        samples['scale'] = None
        if not temperatures is None:
            samples['scale'] = temperatures[epoch]

        with torch.cuda.amp.autocast():
            outputs = model(samples)

        for layer in range(len(linear_probes)):
            for head in range(len(linear_probes[layer])):
                for rep_nr, rep in enumerate(reps):
                    # prepaer outputs
                    if len(outputs[rep].shape) == 4:
                        if rep == 'x_intermed':
                            if head > 0:
                                continue
                            outs = outputs[rep].reshape(
                                outputs[rep].shape[0], args.mnist_deit_depth, -1)
                        else:
                            # depends on patch size
                            outs = outputs[rep].reshape(
                                outputs[rep].shape[0], args.mnist_deit_depth, args.num_heads, 65, 64)
                    else:
                        outs = outputs[rep]

                    for target in range(len(linear_probes[layer][head])):
                        with torch.cuda.amp.autocast():
                            if args.cls_token_linprobe:
                                if rep == 'x_intermed':
                                    outs = outputs[rep][:, layer, 0, :]
                                    probe_out = linear_probes[layer][head][target][rep_nr](
                                        outs.detach())
                                else:
                                    probe_out = linear_probes[layer][head][target][rep_nr](
                                        outs[:, layer, head, 0, :].detach())
                            else:
                                if rep == 'x_intermed':
                                    probe_out = linear_probes[layer][head][target][rep_nr](
                                        outs[:, layer, :].reshape(outputs[rep].shape[0], -1).detach())
                                else:
                                    probe_out = linear_probes[layer][head][target][rep_nr](
                                        outs[:, layer, head, :].reshape(outs.shape[0], -1))
                        loss = criterion(probe_out, targets[:, target].squeeze())
                        acc1 = accuracy(probe_out, targets[:, target], topk=(1,))
                        acc2 = accuracy(probe_out, targets[:, target], topk=(2,))

                        loss_value = loss.item()
                        torch.cuda.synchronize()
                        metric_loggers[layer][head][target][rep_nr].update(loss=loss_value)
                        metric_loggers[layer][head][target][rep_nr].update(acc1=acc1[0])
                        metric_loggers[layer][head][target][rep_nr].update(acc2=acc2[0])

    # log metrics
    lin_probe_result_dict = {}
    for layer in range(len(linear_probes)):
        for head in range(len(linear_probes[layer])):
            for target in range(len(linear_probes[layer][head])):
                for rep_nr, rep in enumerate(reps):
                    if rep == 'x_intermed' and head > 0:
                        continue
                    metric_loggers[layer][head][target][rep_nr].synchronize_between_processes()
                    if epoch is not None:
                        metric_loggers[layer][head][target][rep_nr].log_val(epoch=epoch)
                        metric_loggers[layer][head][target][rep_nr].print_val(epoch=epoch)
                        lin_probe_result_dict[f'layer_{layer}_head_{head}_target_{target_list[target]}_rep_{reps[rep_nr]}'] = {}
                        lin_probe_result_dict[f'layer_{layer}_head_{head}_target_{target_list[target]}_rep_{reps[rep_nr]}']['acc1'] = metric_loggers[layer][head][target][rep_nr].meters['acc1'].global_avg
                        lin_probe_result_dict[f'layer_{layer}_head_{head}_target_{target_list[target]}_rep_{reps[rep_nr]}']['acc2'] = metric_loggers[layer][head][target][rep_nr].meters['acc2'].global_avg
                        lin_probe_result_dict[f'layer_{layer}_head_{head}_target_{target_list[target]}_rep_{reps[rep_nr]}']['loss'] = metric_loggers[layer][head][target][rep_nr].meters['loss'].value
                        lin_probe_result_dict[f'layer_{layer}_head_{head}_target_{target_list[target]}_rep_{reps[rep_nr]}']['nr_samples'] = len(data_loader.dataset)

    np.save(os.path.join(args.output_dir, 
                         f'online_eval_data_subset{args.dataset_subset_fraction}_ep{epoch}.npy'), 
            lin_probe_result_dict, allow_pickle=True)


def log_grad_qkv(map):
    fig = plt.figure(dpi=150)
    plt.imshow(map)
    plt.colorbar()
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=150)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    plt.close()

    return img_arr


def log_attn(img_nr, attn, img, args, labels, has_cls=True):
    attn_all = attn[img_nr, :, :, :] # nr_nheads, ntok, ntok
    idx = 0
    img = img[img_nr].permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1) * 255
    final_img = []

    for layer in range(args.mnist_deit_depth):
        layer_i = []
        res = int(math.sqrt(attn_all[0].shape[0]))
        if has_cls:
            img_canvas = np.zeros((img.shape[0] + int(np.round(1 / res * img.shape[0])),
                                   img.shape[1] + int(np.round(1 / res * img.shape[1])), 3), 
                                   dtype=np.uint8)
            img_canvas[int(np.round(1 / res * img.shape[0])):, 
                       int(np.round(1 / res * img.shape[1])):] = img
        else:
            img_canvas = img
        layer_i.append(img_canvas)

        for head in range(args.num_heads):
            attn = attn_all[idx, :, :]
            full_attn = torch.sum(attn, dim=0)
            if has_cls:
                attn = full_attn[1:].reshape((res, res)) 
                attn = attn / torch.max(full_attn)
                cls_attn = full_attn[0] / torch.max(full_attn)
                attn_image = (attn.numpy() * 255).astype(np.uint8)
                cls_attn = np.uint8(cls_attn * 255)
                canvas = np.zeros((attn_image.shape[0]+1, 
                                   attn_image.shape[1]+1), dtype=np.uint8)
                canvas[1:, 1:] = attn_image
                canvas[0, 0] = cls_attn
            else:
                attn = full_attn.reshape((res, res))
                attn = attn / torch.max(full_attn)
                canvas = (attn.numpy() * 255).astype(np.uint8)
            attn_image = Image.fromarray(canvas)
            attn_image_resized = np.array(attn_image.resize((img_canvas.shape[1], 
                                                             img_canvas.shape[0])))
            layer_i.append(np.ones((5, attn_image_resized.shape[1], 3)).astype(np.uint8)*255)
            layer_i.append(
                np.tile(attn_image_resized.reshape(attn_image_resized.shape[0], 
                attn_image_resized.shape[1], 1), ([1, 1, 3])))
            idx += 1
        layer_i = np.concatenate(layer_i, axis=0)
        final_img.append(layer_i)    
        final_img.append(np.ones((layer_i.shape[0], 5, 3)).astype(np.uint8) * 255)
    
    final_img = np.concatenate(final_img, axis=1)

    return final_img


def evaluate(data_loader, model, device, args, tb_writer=None, epoch=None, temperatures=None):
    if 'resnet' in args.model:
        return evaluate_resnet(data_loader, model, device, args, tb_writer, epoch)
    else:
        return evaluate_deit(data_loader, model, device, args, tb_writer, epoch, temperatures=temperatures)
