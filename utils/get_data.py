"""Contains functions for extracting data into the data folder. 

Not needed for torchvision datasets since they install automatically. 
"""

import torch
from torchvision import datasets
from torchvision.transforms import v2
from pathlib import Path

from config import DATA_DIRECTORY


def get_FashionMNIST():
    """Downloads the FashionMNIST dataset from the Pytorch datasets into the data folder.
    
    Must be called from the main folder so it can find the data folder.
    """
    dir = DATA_DIRECTORY
    #train_data = datasets.FashionMNIST(
    datasets.FashionMNIST(
	    root=dir,	# directory to download the data
        train=True, 	# train or test data
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),	# transformation for the data
        target_transform=None	# transformation for the labels / targets
        )

    #test_data = datasets.FashionMNIST(
    datasets.FashionMNIST(
        root=dir,	# directory to download the data
        train=False, 	# train or test data
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),	# transformation for the data
        target_transform=None	# transformation for the labels / targets
        )
if __name__ == "__main__": 
    get_FashionMNIST()
