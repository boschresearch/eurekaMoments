# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0


import os
import sys
import random
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from matplotlib import colors
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class MNIST_SPACIAL_DECISION(torch.utils.data.Dataset):
    def __init__(self, train, num_colors_per_class=1, args=None,
                 mnist_path="./data/mnist/processed",
                 fashion_mnist_path='./data/fashion_mnist/fashion_mnist.npz',
                 cifar10_path='./data',
                 dataset_subset=1):
        """
            Args
                train (bool): whether to load training or validation data
                num_colors_per_class (int): specify how many different colors to use for each class
                args: namespace with fields:
                 - mnist_task: specifying the task (see description below)
                 - num_mnist_targets: number of targets (can be ignored except for linear probe experiments)
                 - debug: flag to save example images for debugging purposes
                mnist_path: path to mnist
                fashion_mnist_path: path to fashion mnist (only when used for given task)
                cifar10_path: path to cifar 100. Only necessary when used for task

        general setup: indicators indicate which of the 2 targets to classify. Unless stated otherwise they are at
        top left and bottom right.
                        
        ### tasks ###
            spacial_decision_indicator_digit_1_2_fashion
            sd_indicator_digit_1_2_fashion_no_fixed_pos: no_position_task from paper
            mnist_fashion_cifar_ind: if both cifar images are either both cars or both birds classify fashion else classify digit
            fashion_position_as_indicator_topIfAbove: if fashion images on top of each other --> classify top image else bottom
            same_color_distractors_colored_target: 4 digits. if color of indicators same --> top right
            spacial_decision_indicator_digit_1_2_colored_target_or_fashion: digits from same class --> top right fashion classification 
                else top right color classification
            special_decision_digit_group_4_fashion: digits from same set i.e. [1,2] or [3, 4] --> top right fashion classification 
                else bottom left fashion classification

        """
        # load MNIST data
        self.target_samples = None
        self.target_labels = None
        data = torch.load(os.path.join(mnist_path, "training.pt" if train else "test.pt"))
        self.samples = data[0].unsqueeze(1).repeat((1, 3, 1, 1)).float() / 255.0
        self.labels = data[1]
        if not dataset_subset == 1:
            shuffle_idx = np.arange(0,len(self.samples))
            random.shuffle(shuffle_idx)
            shuffle_idx = shuffle_idx[0:int(np.round(dataset_subset*len(self.samples)))]
            self.samples = self.samples[shuffle_idx]
            self.labels = self.labels[shuffle_idx]

        # for particular tasks we use different target data
        if args.mnist_task == 'fashion_position_as_indicator_topIfAbove' or \
                args.mnist_task == 'spacial_decision_indicator_digit_1_2_fashion' or \
                args.mnist_task == 'spacial_decision_indicator_digit_1_2_colored_target_or_fashion' or \
                args.mnist_task == 'sd_indicator_digit_1_2_fashion_no_fixed_pos' or \
                args.mnist_task == 'special_decision_digit_group_fashion' or \
                args.mnist_task == 'special_decision_digit_group_4_fashion':
            fashion_data = np.load(fashion_mnist_path)
            self.target_samples = torch.Tensor(fashion_data[f'x_{"train" if train else "test"}']).unsqueeze(1)
            self.target_samples = self.target_samples.repeat((1, 3, 1, 1)).float() / 255.0
            self.target_labels = torch.Tensor(fashion_data[f'y_{"train" if train else "test"}']).long()
            if not dataset_subset == 1:
                shuffle_idx = np.arange(0,len(self.target_samples))
                random.shuffle(shuffle_idx)
                shuffle_idx = shuffle_idx[0:int(np.round(dataset_subset*len(self.target_samples)))]
                self.target_samples = self.target_samples[shuffle_idx]
                self.target_labels = self.target_labels[shuffle_idx]
        if args.mnist_task == 'mnist_fashion_cifar_ind':
            fashion_data = np.load(fashion_mnist_path)
            self.fasion_target_samples = torch.Tensor(fashion_data[f'x_{"train" if train else "test"}']).unsqueeze(1)
            self.fasion_target_samples = self.fasion_target_samples.repeat((1, 3, 1, 1)).float() / 255.0
            self.fasion_target_labels = torch.Tensor(fashion_data[f'y_{"train" if train else "test"}']).long()
            if not dataset_subset == 1:
                shuffle_idx = np.arange(0,len(self.target_samples))
                random.shuffle(shuffle_idx)
                shuffle_idx = shuffle_idx[0:int(np.round(dataset_subset*len(self.fasion_target_samples)))]
                self.fasion_target_samples = self.fasion_target_samples[shuffle_idx]
                self.fasion_target_labels = self.fasion_target_labels[shuffle_idx]

        if 'cifar' in args.mnist_task:
            self.cifar = CIFAR10(cifar10_path, train=train, transform=None,
                                    target_transform=None, download=False, labelset='fine')
            self.target_samples_cifar = (torch.Tensor(self.cifar.data).float() / 255.0).permute(0, 3, 1, 2)
            self.target_samples_cifar = torch.nn.functional.interpolate(self.target_samples_cifar, (28, 28), mode='bilinear')
            self.target_labels_cifar = torch.tensor(self.cifar.targets)
            if not dataset_subset == 1:
                shuffle_idx = np.arange(0,len(self.target_samples_cifar))
                random.shuffle(shuffle_idx)
                shuffle_idx = shuffle_idx[0:int(np.round(dataset_subset*len(self.target_labels_cifar)))]
                self.target_samples_cifar = self.target_samples_cifar[shuffle_idx]
                self.target_labels_cifar = self.target_labels_cifar[shuffle_idx]                           

        if self.target_samples is None:
            self.target_samples = self.samples
            self.target_labels = self.labels

        self.num_colors_per_class = num_colors_per_class
        self.color_dict = self.get_default_color_dict(args)
        self.color_list = list(self.color_dict.values())
        self.task = args.mnist_task
        self.debug = args.debug
        self.mnist_transform = transforms.Normalize(
            mean=torch.tensor([0.4850, 0.4560, 0.4060]),
            std=torch.tensor([0.2290, 0.2240, 0.2250]))

        if self.task == 'spacial_decision_indicator_digit_1_2_fashion' or \
                self.task == 'spacial_decision_indicator_digit_1_2_colored_target_or_fashion' or \
                self.task == 'sd_indicator_digit_1_2_fashion_no_fixed_pos' or \
                self.task == 'mnist_fashion_cifar_ind':
            labels = np.array([int(lab.numpy()) for lab in self.labels])
            if self.task == 'mnist_fashion_cifar_ind':
                self.indicator_samples = self.target_samples_cifar[
                    np.logical_or(np.array(self.target_labels_cifar == 1), 
                                  np.array(self.target_labels_cifar == 2))]
                self.indicator_labels = self.target_labels_cifar[
                    np.logical_or(np.array(self.target_labels_cifar == 1), 
                                  np.array(self.target_labels_cifar == 2))]
            else:
                self.indicator_samples = self.samples[np.logical_or(np.array(labels == 1), 
                                                                    np.array(labels == 2))]
                self.indicator_labels = self.labels[np.logical_or(np.array(labels == 1), 
                                                                  np.array(labels == 2))]
        elif self.task == 'special_decision_digit_group_4_fashion':
            self.indicator_samples = self.samples[
                np.array(self.labels == 1) | np.array(self.labels == 2) | \
                    np.array(self.labels == 3) | np.array(self.labels == 4)]
            self.indicator_labels = self.labels[
                np.array(self.labels== 1) | np.array(self.labels== 2) | \
                    np.array(self.labels== 3) | np.array(self.labels== 4)]
        else:
            self.indicator_samples = self.samples
            self.indicator_labels = self.labels

        self.nb_classes = 10
        self.class_to_idx = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "9": 8, "10": 9}
        self.color_labels = self.get_color_idx_dict()
        self.color_to_idx = self.color_labels
        self.num_targets = args.num_mnist_targets
        self.linear_probe_training = args.linear_probes_all_heads
        self.cls_token_linprobe = args.cls_token_linprobe

        # used only for linear probe experiments
        if args.num_mnist_targets == 8:
            # digit and color
            self.nb_classes = [len(self.class_to_idx)] * 4 + [len(self.color_labels)] * 4
        if args.num_mnist_targets == 9:
            # digit and color target location
            self.nb_classes = [len(self.class_to_idx)] * 4 + [len(self.color_labels)] * 5
        if args.num_mnist_targets == 6:
            # digit and color target location
            self.nb_classes = [10, 10, 10, 10, 10, 10]

        if not('colored_target' in self.task) :
            self.colored_targets = False
        else:
            self.colored_targets = True

        self.top_right_probability = args.top_right_probability
        if not self.top_right_probability == 0.5:
            self.indi_l1_locs = self.indicator_labels == 1
            self.indi_l2_locs = self.indicator_labels == 2
            self.indi_subset1 = self.indicator_samples[self.indi_l1_locs]
            self.indi_subset2 = self.indicator_samples[self.indi_l2_locs]
            self.indi_subset1_labels = self.indicator_labels[self.indi_l1_locs]
            self.indi_subset2_labels = self.indicator_labels[self.indi_l2_locs]

        self.save_eval_set = args.save_eval_set
        self.eval_set_save_path = args.eval_set_save_path
        self.is_train = train
        self.use_saved_eval_set = args.use_saved_eval_set

    def __getitem__(self, index):
        samples = []
        labels = []

        if not self.is_train and not self.eval_set_save_path is None and self.use_saved_eval_set:
            samp = torch.load(os.path.join(self.eval_set_save_path, f'{index}.pt'))
            sample = samp['tensor']
            label = samp['label']
            return sample, label

        # get 3 additional random samples. use index for top right sample
        for i in range(4):
            if i == 0 or i == 3:
                samples, labels = self.__get_indicators__(i, samples, labels)
            elif i == 1 or i == 2:
                samples, labels = self.__get_targets__(i, index, samples, labels)
        samples, labels, color_idxs = self.__apply_colors_and_color_noise__(samples, labels)
        # now figure out which label is the correct one
        samples, is_top_right_target, labels, label = self.__apply_rule__(samples, labels, color_idxs)
        # concatenate samples. For top_if_above determine correct label
        sample, label, is_top_right_target = self.__concat_samples__(samples, labels, label, is_top_right_target)
        # labels for linear probing experiments
        if self.linear_probe_training:
            label = self.__get_linear_probe_labels__(is_top_right_target, label, labels, color_idxs)

        if self.debug:
            vis_sample = torch.clip(sample.permute(1, 2, 0), 0, 1).numpy() * 255
            vis_sample = vis_sample.astype(np.uint8)
            img = Image.fromarray(vis_sample)
            out_folder = './debug_imgs'
            os.makedirs(out_folder, exist_ok=True)
            img.save(os.path.join(out_folder, str(index)+'_label_'+str(label)+'.jpg'))
            exit()
        if self.save_eval_set and not self.is_train:
            os.makedirs(self.eval_set_save_path, exists_ok=True)
            torch.save({'tensor': sample, 'label': label}, os.path.join(self.eval_set_save_path, f'{index}.pt'))

        return sample, label

    def __get_indicators__(self, i, samples, labels):
        r_idx = np.random.randint(0, len(self.indicator_samples))
        if self.top_right_probability == 0.5 or i == 0:
            samples.append(self.indicator_samples[r_idx])
            labels.append(self.indicator_labels[r_idx])
        elif not self.top_right_probability == 0.5:
            # pick the same label with p=self.top_right_probability
            if np.random.uniform(0, 1) < self.top_right_probability:
                if labels[0] == 1:
                    r_idx = np.random.randint(0, len(self.indi_subset1))
                    samples.append(self.indi_subset1[r_idx])
                    labels.append(self.indi_subset1_labels[r_idx])
                elif labels[0] == 2:
                    r_idx = np.random.randint(0, len(self.indi_subset2))
                    samples.append(self.indi_subset2[r_idx])
                    labels.append(self.indi_subset2_labels[r_idx])
            else:
                samples.append(self.indicator_samples[r_idx])
                labels.append(self.indicator_labels[r_idx])

        return samples, labels

    def __get_targets__(self, i, index, samples, labels):
        if not self.task == 'mnist_fashion_cifar_ind':
            target_idx = index if i == 1 else np.random.randint(0, len(self.target_samples), (1,))[0]
            samples.append(self.target_samples[target_idx])
            labels.append(self.target_labels[target_idx])
        else:
            if i == 1:
                target_idx = np.random.randint(0, len(self.target_labels), (1,))[0]
                samples.append(self.target_samples[target_idx])
                labels.append(self.target_labels[target_idx])
            else:
                target_idx = np.random.randint(0, len(self.fasion_target_samples), (1,))[0]
                samples.append(self.fasion_target_samples[target_idx])
                labels.append(self.fasion_target_labels[target_idx])  

        return samples, labels

    def __apply_colors_and_color_noise__(self, samples, labels):
        color_names = []
        if self.task == 'mnist_fashion_cifar_ind':
            # sort to normal order, s.t. you can apply all transformations as usual
            samples = [samples[s] for s in [1, 0, 3, 2]]
            labels = [labels[s] for s in [1, 0, 3, 2]]
        color_idxs = self.get_random_colors()
        for i in range(4):
            color_names.append(self.color_dict[color_idxs[i]][0])
            color_rgb = np.asarray(colors.hex2color(color_names[i]))
            # add a small amount of color noise to get slight color variations
            color_noise = np.random.normal(loc=0.0, scale=0.05, size=3)
            # color only for indicators
            if (i == 0 or i == 3) or self.colored_targets:
                color_rgb = color_rgb + color_noise
            else:
                color_rgb = np.asarray(colors.hex2color('white')) + color_noise
            if color_rgb is not None:
                color_rgb_tensor = torch.tensor(color_rgb).unsqueeze(-1).unsqueeze(-1).float()
                # multiply with image to give it color
                samples[i] = samples[i] * color_rgb_tensor
        if self.task == 'mnist_fashion_cifar_ind':
            # sort back
            samples = [samples[s] for s in [1, 0, 3, 2]]
            labels = [labels[s] for s in [1, 0, 3, 2]]    

        return samples, labels, color_idxs

    def __apply_rule__(self, samples, labels, color_idxs):
        conditions = [self.__get_conditions__(color_idxs[0], labels[0]),
                      self.__get_conditions__(color_idxs[3], labels[3])]
        if self.task == 'fashion_position_as_indicator_topIfAbove' or \
                self.task == 'same_color_distractors_colored_target':
            # top right if distractors have same color
            if conditions[0][:2] == conditions[1][:2]:
                is_top_right_target = True
            else:
                is_top_right_target = False
        elif self.task == 'spacial_decision_indicator_digit_1_2_fashion' or \
                self.task == 'sd_indicator_digit_1_2_fashion_no_fixed_pos' or \
                self.task == 'mnist_fashion_cifar_ind' or \
                self.task == 'spacial_decision_indicator_digit_1_2_colored_target_or_fashion':
            if labels[0] == labels[3]:
                is_top_right_target = True
            else:
                is_top_right_target = False
                if self.task == 'spacial_decision_indicator_digit_1_2_colored_target_or_fashion':
                    # replace label with color index, since hti si the task now
                    labels[2] = torch.as_tensor(color_idxs[1])  
        elif self.task == 'special_decision_digit_group_4_fashion':
            if (labels[0] < 3 and labels[3] < 3) or (labels[0] >= 3 and labels[3] >= 3):
                is_top_right_target = True
            else:
                is_top_right_target = False

        if is_top_right_target:
            label = labels[1]
        else:
            label = labels[2]

        return samples, is_top_right_target, labels, label

    def __concat_samples__(self, samples, labels, label, is_top_right_target):
        # Labels do not have to be permuted, since relevant has already been selected
        if not self.task == 'fashion_position_as_indicator_topIfAbove' and \
                not self.task == 'sd_indicator_digit_1_2_fashion_no_fixed_pos' and \
                not self.task == 'mnist_fashion_cifar_ind':
            sample_top = torch.cat([samples[0], samples[1]], dim=2)
            sample_bot = torch.cat([samples[2], samples[3]], dim=2)
        elif self.task == 'fashion_position_as_indicator_topIfAbove':
            if random.randint(0, 1) > 0:  # above each other?
                label = labels[1]
                if random.randint(0, 1) > 0:  # left?
                    sample_top = torch.cat([samples[1], torch.zeros_like(samples[1])], dim=2)
                    sample_bot = torch.cat([samples[2], torch.zeros_like(samples[2])], dim=2)
                else:
                    sample_top = torch.cat([torch.zeros_like(samples[1]), samples[1]], dim=2)
                    sample_bot = torch.cat([torch.zeros_like(samples[2]), samples[2]], dim=2)
            else:
                label = labels[2]
                if random.randint(0, 1) > 0:  # left?
                    sample_top = torch.cat([samples[1], torch.zeros_like(samples[1])], dim=2)
                    sample_bot = torch.cat([torch.zeros_like(samples[2]), samples[2]], dim=2)
                else:
                    sample_top = torch.cat([torch.zeros_like(samples[1]), samples[1]], dim=2)
                    sample_bot = torch.cat([samples[2], torch.zeros_like(samples[2])], dim=2)
        elif self.task == 'sd_indicator_digit_1_2_fashion_no_fixed_pos':
            if random.randint(0, 1) > 0:
                sample_top = torch.cat([samples[0], samples[1]], dim=2)
            else:
                sample_top = torch.cat([samples[1], samples[0]], dim=2)
            if random.randint(0, 1) > 0:
                sample_bot = torch.cat([samples[2], samples[3]], dim=2)
            else:
                sample_bot = torch.cat([samples[3], samples[2]], dim=2)
        elif self.task == 'mnist_fashion_cifar_ind':
            np.random.shuffle(samples)
            sample_top = torch.cat([samples[0], samples[1]], dim=2)
            sample_bot = torch.cat([samples[2], samples[3]], dim=2)
        sample = torch.cat([sample_top, sample_bot], dim=1)

        return sample, label, is_top_right_target

    def __get_linear_probe_labels__(self, is_top_right_target, label, labels, color_idxs):
        if self.num_targets == 1:
            label = np.array([int(is_top_right_target)])
        if self.num_targets == 8:
            label = torch.tensor(np.concatenate([np.array([l.numpy() for l in labels]), 
                                                (color_idxs <= 4).astype(int)]))
            label = label.unsqueeze(1)
        if self.num_targets == 9:
            label = torch.tensor(np.concatenate([np.array([l.numpy() for l in labels]), 
                                                 (color_idxs <= 4).astype(int),
                                                 np.array([int(is_top_right_target)])]))
            label = label.unsqueeze(1)
        if self.num_targets == 6:
            label = torch.tensor(np.concatenate([np.array([l.numpy() for l in labels]),
                                                 np.array([int(is_top_right_target)]),
                                                 np.array([label])]))
            label = label.unsqueeze(1)
        return label

    def get_default_color_dict(self, args=None):
        # returns a hard-coded pre-defined color-digit-correlation dict
        if args.mnist_task == 'spacial_decision_indicator_digit_1_2_colored_target_or_fashion':
            return {0: ["brown"], 1: ["blue"], 2: ["yellow"], 3: ["orange"], 4: ["red"], 5: ["green"],
                    6: ["purple"], 7: ["gray"], 8: ["pink"], 9: ["turquoise"]}
        else:
            return {0: ["blue"], 1: ["blue"], 2: ["blue"], 3: ["blue"], 4: ["blue"], 5: ["red"],
                    6: ["red"], 7: ["red"], 8: ["red"], 9: ["red"]}
        
    def get_color_idx_dict(self):
        colors = sum([c[:self.num_colors_per_class] for c in self.color_dict.values()], [])
        color_labels = {}
        for i, c in enumerate(colors):
            color_labels[c] = i
        return color_labels


    def __get_conditions__(self, color_idx, label):
        condition = ''
        if color_idx < (len(self.color_list) / 2):
            condition = condition + 'c1'
        else:
            condition = condition + 'c2'

        if label < (len(self.color_list) / 2):
            condition = condition + 'd1'
        else:
            condition = condition + 'd2'
        return condition

    def get_random_colors(self):
        if self.top_right_probability == 0.5 or self.task == 'spacial_decision_indicator_digit_1_2_fashion':
            return np.random.randint(0, len(self.color_list), (4,))
        else:
            colors = np.random.randint(0, len(self.color_list), (4,))
            if np.random.random() < self.top_right_probability:
                colors[3] = colors[0] 
            else:
                colors[3]= (len(self.color_list) - 1) - colors[0]
            return colors

    def __len__(self):
        return self.target_samples.shape[0]


def build_dataset(is_train, args, data_path=None):
    ds = args.data_set
    if not is_train and not args.valid_data_set is None:
        ds = args.valid_data_set
    if ds.lower() == 'mnist_spacial_decision':
        dataset = MNIST_SPACIAL_DECISION(train=is_train, num_colors_per_class=args.colors_per_class,
                        args=args, dataset_subset=args.dataset_subset_fraction)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes

# The following function is code from 
#   (https://github.com/facebookresearch/deit/)
# and was published under  Apache 2.0 license
# Copyright (c)  Meta Platforms, Inc. and affiliates. 2015-present, Facebook, Inc.
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)

        return transform

    t = []
    if resize_im:
        size = int((1.0) * args.input_size) 
        t.append(
            transforms.Resize(size, interpolation=3),
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)



# The following class contain modified code from 
#   (https://github.com/pytorch/vision/blob/main/torchvision)
# Copyright (c) Soumith Chintala 2016, 
# All rights reserved.
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, labelset='fine'):

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry and labelset == 'fine':
                    self.targets.extend(entry['labels'])
                elif labelset == 'both':
                    self.targets.append([entry['coarse_labels'], entry['fine_labels']])
                elif labelset == 'coarse':
                    self.targets.extend(entry['coarse_labels'])
                elif labelset == 'filenames':
                    self.targets.extend(entry['filenames'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC


        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if not len(self.targets) == 1:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], [self.targets[0][0][index], self.targets[0][1][index]]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
