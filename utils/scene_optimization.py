"""Optional training controls; zero warmup and zero EMA retain the old path."""
import copy
import math
import torch


def lr_multiplier(epoch, epochs, warmup, minimum):
    if epoch < warmup:
        return (epoch + 1) / warmup
    return minimum + (1 - minimum) * (1 + math.cos(math.pi * min(epoch - warmup, epochs - warmup) / (epochs - warmup))) / 2


class ModelEMA:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        for key, value in self.model.state_dict().items():
            source = model.state_dict()[key].detach()
            if value.is_floating_point():
                value.lerp_(source, 1 - self.decay)
            else:
                value.copy_(source)
