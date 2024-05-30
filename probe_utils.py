# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch.nn as nn
from timm_future_imports import Mlp

class LinearProbeLayer(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.lin_layer = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.lin_layer(x)

class LinearProbeCollection():
    def __init__(self, model, nb_classes, ntargets, args):
        if 'deit' in args.model:
            self.nfeat = 4160
        elif args.cls_token_linprobe:
            self.nfeat = args.embedding_dim_per_head
        elif 'vanilla' in args.model:
            self.nfeat = args.embedding_dim_per_head * 197
        else:
            self.nfeat = args.embedding_dim_per_head * 196
        self.depth = model.depth

        self.num_heads = model.num_heads
        self.ntargets = ntargets
        if isinstance(nb_classes, int):
            nb_classes = [nb_classes]

        if args.return_qkv:
            self.num_reps = 4
        else:
            self.num_reps = 1
        if args.return_intermed_x:
            self.num_reps += 1
        self.mlp_probes = args.mlp_probes

        probes = []
        for d in range(self.depth):
            d_probes = []
            for h in range(self.num_heads):
                h_probes = []
                for target in range(self.ntargets):
                    r_probes = []
                    for rep in range(self.num_reps):
                        if not args.mlp_probes:
                            if rep == self.num_reps-1 and args.return_intermed_x:
                                r_probes.append(LinearProbeLayer(self.nfeat * args.num_heads, nb_classes[target]))
                            else:
                                r_probes.append(LinearProbeLayer(self.nfeat, nb_classes[target]))
                        else:
                            if rep == self.num_reps-1 and args.return_intermed_x:
                                r_probes.append(Mlp(self.nfeat * args.num_heads, 10,  nb_classes[target]))
                            else:
                                r_probes.append(Mlp(self.nfeat, 10, nb_classes[target]))
                    h_probes.append(r_probes)
                d_probes.append(h_probes)
            probes.append(d_probes)

        self.probe_list = probes

    def to_gpu(self):
        for d in range(self.depth):
            for h in range(self.num_heads):
                for t in range(self.ntargets):
                    for r in range(self.num_reps):
                        if not self.mlp_probes:
                            self.probe_list[d][h][t][r].lin_layer.cuda()
                        else:
                            self.probe_list[d][h][t][r].cuda()

    def set_trainable(self):
        for d in range(self.depth):
            for h in range(self.num_heads):
                for t in range(self.ntargets):
                    for r in range(self.num_reps):
                        if not self.mlp_probes:
                            self.probe_list[d][h][t][r].lin_layer.requires_grad = True
                        else:
                            for n, v in self.probe_list[0][0][0][0].named_parameters():
                                v.requires_grad = True

    def train(self):
        for d in range(self.depth):
            for h in range(self.num_heads):
                for t in range(self.ntargets):
                    for r in range(self.num_reps):
                        if not self.mlp_probes:
                            self.probe_list[d][h][t][r].lin_layer.train(True)
                        else:
                            self.probe_list[d][h][t][r].train(True)
    def eval(self):
        for d in range(self.depth):
            for h in range(self.num_heads):
                for t in range(self.ntargets):
                    for r in range(self.num_reps):
                        self.probe_list[d][h][t][r].eval()

    def get_optimizers(self, args, create_optimizer_fn, **kwargs):
        optims = []
        for d in range(self.depth):
            d_optims = []
            for h in range(self.num_heads):
                h_optims = []
                for t in range(self.ntargets):
                    r_optims = []
                    for r in range(self.num_reps):
                        r_optims.append(create_optimizer_fn(args, self.probe_list[d][h][t][r], kwargs))
                    h_optims.append(r_optims)
                d_optims.append(h_optims)
            optims.append(d_optims)

        return optims