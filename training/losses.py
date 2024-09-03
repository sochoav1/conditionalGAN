import torch
import torch.nn as nn


class GANLoss:
    def __init__(self, device):
        self.device = device
        self.loss = nn.BCELoss()
        self.real_label = 1.0
        self.fake_label = 0.0

    def get_labels(self, predictions, is_real):
        if is_real:
            return torch.full(predictions.shape, self.real_label, device=self.device)
        else:
            return torch.full(predictions.shape, self.fake_label, device=self.device)

    def __call__(self, predictions, is_real):
        labels = self.get_labels(predictions, is_real)
        return self.loss(predictions, labels)
    
class L1Loss:
    def __init__(self):
        self.loss = nn.L1Loss()

    def __call__(self, generated, real):
        return self.loss(generated, real)