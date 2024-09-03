import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            # Encoder
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Decoder
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, output_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Permute input to (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = self.model(x)
        # Permute output back to (N, H, W, C)
        return x.permute(0, 2, 3, 1)