import os
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super().__init__()

        # classes
        self.num_classes = 5

        # nano model
        self.depth = 0.33
        self.width = 0.25

        # experiment name
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # dataset path
        self.data_dir = "datasets/prepared/formulas_coco_v1"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        # input size
        self.input_size = (416, 416)
        self.test_size = (416, 416)

        # epochs
        self.max_epoch = 150
        self.no_aug_epochs = 10
        self.warmup_epochs = 3

        # adapt batch and lr to the CPU/GPU environment