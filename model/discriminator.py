import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, num_channels):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(num_channels * 2, 64, 4, 2, 1),
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
            nn.Conv2d(512, 1, 4, 1, 0),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, condition):
        img = img.permute(0, 3, 1, 2)
        condition = condition.permute(0, 3, 1, 2)
        
        x = torch.cat([img, condition], dim=1)
        x = self.model(x)
        return self.sigmoid(x.squeeze())