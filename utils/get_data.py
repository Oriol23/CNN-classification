"""Contains functions for extracting data into the data folder. 

"""
import os
import torch
from torchvision import datasets
from torchvision.transforms import v2

from config import DATA_DIRECTORY

def get_FashionMNIST(dataset_name="FashionMNIST"):
    """Downloads the FashionMNIST dataset from the Pytorch datasets into the data folder.

        The folder structure of a dataset will be ...data/dataset_name/FashionMNIST/raw

    Args: 
        dataset_name: the name of the dataset. 
    """
    dataset_name = r'{}'.format(dataset_name)
    dir = os.path.join(DATA_DIRECTORY,dataset_name)
    if os.path.exists(dir):
        print(f"Dataset {dataset_name} already exists, will not rewrite.")
    datasets.FashionMNIST(
	    root=dir,	
        train=True, 	# train or test data
        download=True,
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),	
        target_transform=None	
        )

    datasets.FashionMNIST(
        root=dir,	
        train=False, 	
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),	
        target_transform=None	
        )


if __name__ == "__main__": 
    get_FashionMNIST()
