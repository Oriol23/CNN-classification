"""Contains functions for extracting a torchvision dataset into the data folder of the project. 

"""
import os
from torch import float32
from torchvision import datasets
from torchvision.transforms import v2

from config import DATA_DIRECTORY

def get_FashionMNIST(dataset_name="FashionMNIST"):
    """Downloads the PyTorch FashionMNIST dataset into the /data folder.

    The folder structure of a dataset will be data/dataset_name/FashionMNIST/raw

    Args: 
        dataset_name: The name the dataset will be saved under. 
    """
    dir = os.path.normpath(os.path.join(DATA_DIRECTORY,dataset_name))
    if os.path.exists(dir):
        print(f"Dataset {dataset_name} already exists, will not rewrite.")
    datasets.FashionMNIST(
	    root=dir,	
        train=True, 	# train or test data
        download=True,
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(float32, scale=True)]),	
        target_transform=None	
        )

    datasets.FashionMNIST(
        root=dir,	
        train=False, 	
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(float32, scale=True)]),	
        target_transform=None	
        )


#if __name__ == "__main__": 
#    get_FashionMNIST("some spaces are good")
