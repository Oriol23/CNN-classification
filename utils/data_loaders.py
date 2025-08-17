"""Contains functions for creating dataloaders from a given datatset in the data folder."""
import torch 
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from config import DATA_DIRECTORY


data_directory = DATA_DIRECTORY

def create_train_test_dataloaders(downloads=True):
    """Creates train and test dataloaders from the torchvision FashionMNIST dataset.
    
    When run for the first time downloads the data in the data folder. 
    
    Args: 
        downloads: True if you want to download the data"""
    train_data = datasets.FashionMNIST(
        root=data_directory,	# directory to download the data
        train=True, 	# train or test data
        download=downloads,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),	# transformation for the data
        target_transform=None	# transformation for the labels / targets
        )

    test_data = datasets.FashionMNIST(
        root=data_directory,	# directory to download the data
        train=False, 	# train or test data
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),	# transformation for the data
        target_transform=None	# transformation for the labels / targets
        )

    train_dataloader = DataLoader(dataset=train_data, 
                                batch_size=32,
                                shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, 
                                batch_size=32,
                                shuffle=False) # no need to shuffle since test data is not being used to train and order does not matter
    return train_dataloader,test_dataloader