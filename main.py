import torch

from data.data_loader import get_dataloader
from model.conditional_gan import ConditionalGAN
from training.trainer import Trainer
from utils.config import load_config


def main():
    # Load configuration
    config = load_config('experiments/experiment_configs/default_config.yaml')
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize dataloader
    dataloader = get_dataloader(config)
    
    # Initialize models
    cgan = ConditionalGAN(config).to(device)
    
    # Initialize trainer
    trainer = Trainer(cgan, dataloader, config, device)
    
    # Start training
    trainer.train()

if __name__ == '__main__':
    main()