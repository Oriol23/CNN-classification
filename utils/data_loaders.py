"""Contains functions for creating dataloaders from a given datatset in the data folder."""
import os
import sys
import torch 
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from config import DATA_DIRECTORY

simple_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

def create_train_test_dataloaders(dataset_name, transform=simple_transform):
    """Creates train and test dataloaders from the torchvision FashionMNIST dataset.
        
    Args: 
        dataset_name: name of the dataset in the data folder
        transform: transformation applied to the train dataloader. Transformation to tensor is 
        already included.
    """
    
    dataset_name = r'{}'.format(dataset_name)
    data_directory = os.path.join(DATA_DIRECTORY,dataset_name)

    try: 
        train_data = datasets.FashionMNIST(
            root=data_directory,	# directory to download the data
            train=True, 	# train or test data
            download=False,
            transform=v2.Compose([transform,simple_transform]),
            target_transform=None	
            )

        test_data = datasets.FashionMNIST(
            root=data_directory,	# directory to download the data
            train=False, 	# train or test data
            download=False,
            transform=simple_transform, # no need to transform test data
            target_transform=None
            )

        train_dataloader = DataLoader(dataset=train_data, 
                                    batch_size=32,
                                    shuffle=True)
        test_dataloader = DataLoader(dataset=test_data, 
                                    batch_size=32,
                                    shuffle=False) 
        return train_dataloader,test_dataloader
    except RuntimeError: #RuntimeError when PyTorch cannot find the dataset
        print("\n The dataset name does not correspond to any dataset. \n")
        sys.exit(1)


#if __name__ == "__main__": 
#    train,test = create_train_test_dataloaders("FashionMNIST nope") 
#    print(len(train))