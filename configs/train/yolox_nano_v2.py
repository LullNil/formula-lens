import os

import torch.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        annotations_dir = os.path.join(project_root, "datasets", "prepared", "formulas_coco_v2", "annotations")

        self.num_classes = 7
        self.class_names = (
            "block",
            "denominator",
            "exponent",
            "numerator",
            "system_row",
            "text",
            "whole_part",
        )

        self.depth = 0.33
        self.width = 0.25
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.mosaic_prob = 0.5
        self.enable_mixup = False

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_dir = os.path.join(project_root, "datasets", "prepared", "formulas_coco_v2")
        self.train_ann = "instances_train2017.json"
        has_val_split = os.path.exists(os.path.join(annotations_dir, "instances_val2017.json"))
        self.val_ann = "instances_val2017.json" if has_val_split else self.train_ann
        self.eval_split_name = "val2017" if has_val_split else "train2017"
        self.output_dir = os.path.join(project_root, "weights", "finetuned")

        self.input_size = (416, 416)
        self.test_size = (416, 416)

        self.max_epoch = 150
        self.no_aug_epochs = 10
        self.warmup_epochs = 3
        self.data_num_workers = 2
        self.eval_interval = 10
        self.print_interval = 20
        self.save_history_ckpt = True
        # Fine-tuning from v1 checkpoint: lower LR to preserve prior knowledge.
        self.basic_lr_per_img = 0.005 / 64.0
        self.min_lr_ratio = 0.05
        self.mixup_prob = 0.1
        self.hsv_prob = 1.0
        self.flip_prob = 0.5

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

        def init_yolo(module):
            for layer in module.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eps = 1e-3
                    layer.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth,
                self.width,
                in_channels=in_channels,
                act=self.act,
                depthwise=True,
            )
            head = YOLOXHead(
                self.num_classes,
                self.width,
                in_channels=in_channels,
                act=self.act,
                depthwise=True,
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform

        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name=self.eval_split_name if not testdev else "test2017",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )
