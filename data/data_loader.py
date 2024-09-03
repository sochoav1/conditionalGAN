from torch.utils.data import DataLoader

from .dataset import ConditionalGANDataset


def get_dataloader(config):
    dataset = ConditionalGANDataset(
        config['data_path_G'],
        config['data_path_R']
    )
    return DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )