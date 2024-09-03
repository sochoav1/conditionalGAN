import torch
from tqdm import tqdm

from .losses import GANLoss, L1Loss


class Trainer:
    def __init__(self, cgan, dataloader, config, device):
        self.cgan = cgan
        self.dataloader = dataloader
        self.config = config
        self.device = device

        self.gan_loss = GANLoss(device)
        self.l1_loss = L1Loss()

        self.g_optimizer = torch.optim.Adam(self.cgan.generator.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))
        self.d_optimizer = torch.optim.Adam(self.cgan.discriminator.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))

    def train(self):
        for epoch in range(self.config['num_epochs']):
            self.train_epoch(epoch)

    def train_epoch(self, epoch):
        self.cgan.train()
        loop = tqdm(self.dataloader, leave=True)
        for i, (real_G, real_R) in enumerate(loop):
            real_G, real_R = real_G.to(self.device), real_R.to(self.device)
            
            # Train Discriminator
            self.d_optimizer.zero_grad()
            
            # Real images
            d_real = self.cgan.discriminate(real_G, real_R)
            print(f"d_real shape: {d_real.shape}")  # Debug print
            d_real_loss = self.gan_loss(d_real, True)
            
            # Fake images
            fake_G = self.cgan.generate(real_R)
            d_fake = self.cgan.discriminate(fake_G.detach(), real_R)
            print(f"d_fake shape: {d_fake.shape}")  # Debug print
            d_fake_loss = self.gan_loss(d_fake, False)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            self.d_optimizer.step()
            
            # Train Generator
            self.g_optimizer.zero_grad()
            
            # Generate fake images and compute GAN loss
            fake_G = self.cgan.generate(real_R)
            d_fake = self.cgan.discriminate(fake_G, real_R)
            g_gan_loss = self.gan_loss(d_fake, True)
            
            # Compute L1 loss
            g_l1_loss = self.l1_loss(fake_G, real_G) * self.config['lambda_l1']
            
            # Total generator loss
            g_loss = g_gan_loss + g_l1_loss
            g_loss.backward()
            self.g_optimizer.step()
            
            # Update progress bar
            loop.set_postfix(
                D_real=d_real_loss.item(),
                D_fake=d_fake_loss.item(),
                G_gan=g_gan_loss.item(),
                G_L1=g_l1_loss.item(),
            )

    def save_model(self, epoch):
        torch.save({
            'generator': self.cgan.generator.state_dict(),
            'discriminator': self.cgan.discriminator.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
        }, f"checkpoints/model_epoch_{epoch+1}.pth")