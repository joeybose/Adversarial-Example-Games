
from collections import defaultdict
import torch
import warnings

class Logger:
    def __init__(self, *args, **kwargs):
        from expviz.logger import Logger as ExpvizLogger
        self.expviz = ExpvizLogger(*args, **kwargs)

    def write(self, scalar_dict, epoch):
        for key, value in scalar_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.expviz.add_scalar(key, value, epoch)
