import torch
import torchvision.utils as vutils


def save_image_grid(real_R, real_G, fake_G, filename):
    img_grid = torch.cat((real_R, real_G, fake_G), dim=3)
    vutils.save_image(img_grid, filename, normalize=True, nrow=8)