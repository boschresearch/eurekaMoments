# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
import torch
import vit_timm0412 as vanilla_vit
import models
import utils

def model_lib(args, nb_classes):
    if args.model == 'resnet':
        from torchvision.models import resnet18
        model = resnet18(num_classes=nb_classes)
    elif args.model == 'resnet34':
        from torchvision.models import resnet34
        model = resnet34(num_classes=nb_classes) 
    elif args.model == 'resnet9':
        import resnet_custom as resnet
        if args.mnist_task == 'fashion_target_same_color_distractors_neutral_target_no_fixed_position_6channel':
            in_channels = 6
        else:
            in_channels = 3
        model = resnet._resnet(resnet.BasicBlock, [1, 1, 1, 1], None, True, 
                               num_classes=nb_classes, in_channels=in_channels)
    elif args.model == 'vanilla_vit_mnist_patch7_56': 
        if args.mlp_ratio is None:
            args.mlp_ratio = 4
        if args.qkv_bias is None:
            args.qkv_bias = False
        if args.pos_embed_type is None:
            args.pos_embed_type = 'learnable'
        model = vanilla_vit.deit_mnist_patch7_28(
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            img_size=56,
            num_heads=args.num_heads,
            return_z=args.return_z,
            args=args,
            embedding_dim_per_head=args.embedding_dim_per_head,
            mnist_deit_depth=args.mnist_deit_depth,
            mlp_ratio=args.mlp_ratio,
            qkv_bias=args.qkv_bias,
            pos_embed_type=args.pos_embed_type,
            attn_drop_rate=args.attn_drop_rate,
        )
    elif args.model == 'deit_small_patch16_224':
        model = models.deit_small_patch16_224(
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            img_size=args.input_size,
            num_heads=args.num_heads,
            return_z=args.return_z,
            args=args
        )
    return model

# The following function is from deit
# (https://github.com/facebookresearch/deit)
# Copyright (c) 2015-present, Facebook, Inc. Licensed under Apache 2.0 license.
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
def load_checkpoint(args, model):
    if args.pretrain:
        checkpoint = torch.load(args.pretrain, map_location='cpu')
        checkpoint_model = checkpoint['model']
        model.load_state_dict(checkpoint_model, strict=True)

    head_params = None
    if args.finetune or args.lin_eval:
        if args.lin_eval and not args.finetune:
            model_checkpoint = args.lin_eval
        else:
            model_checkpoint = args.finetune
        if model_checkpoint.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                model_checkpoint, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(model_checkpoint, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        head_params = ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']
        if 'resnet' in args.model:
            head_params = ['fc.weight', 'fc.bias']
        if args.dropout_head > 0:
            head_params = [p.replace(".", ".1.") for p in head_params]
        for k in head_params:
            if k in checkpoint_model and (checkpoint_model[k].shape != state_dict[k].shape or 'resnet' in args.model): 
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        if not 'resnet' in args.model:
            # interpolate position embedding
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    return model, head_params


# The following function is from deit
# (https://github.com/facebookresearch/deit)
# Copyright (c) 2015-present, Facebook, Inc. Licensed under Apache 2.0 license.
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
def load_checkpoint_to_resume_training(args, model_without_ddp, optimizer, lr_scheduler, model_ema, loss_scaler):
    if args.resume.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.resume, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.resume, map_location='cpu')
    try:
        if (args.eval):
            msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            print(msg)
            if not all([True for k in msg[0] if 'reg_mlps' in k]):
                raise Exception('missing params that are not just the regularizer')
        else:
            msg = model_without_ddp.load_state_dict(checkpoint['model'])
            print(msg)
    except:
        model_without_ddp.load_state_dict(checkpoint)
    if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.model_ema:
            utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])

    lr_scheduler.step(args.start_epoch)

    return args, model_without_ddp, optimizer, lr_scheduler, model_ema, loss_scaler