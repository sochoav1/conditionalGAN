import torch.nn as nn

from .discriminator import Discriminator
from .generator import Generator


class ConditionalGAN(nn.Module):
    def __init__(self, config):
        super(ConditionalGAN, self).__init__()
        self.num_channels = config['num_channels']

        self.generator = Generator(self.num_channels, self.num_channels)
        self.discriminator = Discriminator(self.num_channels)

    def generate(self, condition):
        return self.generator(condition)

    def discriminate(self, img, condition):
        return self.discriminator(img, condition)