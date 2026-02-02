#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super().__init__()
        # Dataset setup
        self.depth = 1.0
        self.width = 0.75
        self.num_classes = 1  # 'fly'
        self.data_dir = "datasets/drosophila"
        self.train_ann = "annotations/drosophila_train.json"
        self.val_ann = "annotations/drosophila_val.json"
        self.train_img_folder = "images/train"
        self.val_img_folder = "images/val"

        # Image and training settings
        self.input_size = (960, 960)
        self.test_size = (960, 960)
        self.max_epoch = 100
        self.data_num_workers = 4
        self.eval_interval = 10
        self.print_interval = 20
