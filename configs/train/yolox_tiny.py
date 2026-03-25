import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        annotations_dir = os.path.join(project_root, "datasets", "prepared", "formulas_coco_v1", "annotations")

        self.num_classes = 6
        self.class_names = (
            "block",
            "denominator",
            "exponent",
            "numerator",
            "system_row",
            "text",
        )

        self.depth = 0.33
        self.width = 0.375

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_dir = os.path.join(project_root, "datasets", "prepared", "formulas_coco_v1")
        self.train_ann = "instances_train2017.json"
        has_val_split = os.path.exists(os.path.join(annotations_dir, "instances_val2017.json"))
        self.val_ann = "instances_val2017.json" if has_val_split else self.train_ann
        self.eval_split_name = "val2017" if has_val_split else "train2017"
        self.output_dir = os.path.join(project_root, "weights", "finetuned")

        self.input_size = (640, 640)
        self.test_size = (640, 640)

        self.max_epoch = 150
        self.no_aug_epochs = 10
        self.warmup_epochs = 3
        self.data_num_workers = 2
        self.eval_interval = 10
        self.print_interval = 20
        self.save_history_ckpt = True
        self.basic_lr_per_img = 0.01 / 64.0
        self.min_lr_ratio = 0.05
        self.mosaic_prob = 1.0
        self.mixup_prob = 0.1
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.enable_mixup = True
