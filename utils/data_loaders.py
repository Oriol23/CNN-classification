"""Creates dataloaders from a torchvision dataset.

The dataset has to be previously downloaded on the /data folder of the project.
"""
import os
import sys
import torch 
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from config import DATA_DIRECTORY

simple_transform = v2.Compose([v2.ToImage(), 
                               v2.ToDtype(torch.float32, scale=True)])

def create_train_test_dataloaders(dataset_name="FashionMNIST", 
                                  BATCH_SIZE=32, 
                                  transform=None):
    """Creates train and test dataloaders from a torchvision dataset.
        
    Args: 
        dataset_name: Name of the dataset that will be used
        BATCH_SIZE: Batch size of the dataloader
        transform: Transformation applied to the train dataloader. 
            Transformation to tensor is already included.
    """                             
    
    dataset_dir = os.path.normpath(os.path.join(DATA_DIRECTORY,dataset_name))
    if transform == None: 
        transform = simple_transform
    
    try: 
        train_data = datasets.FashionMNIST(
            root=dataset_dir,	# directory to download the data
            train=True, 	# train or test data
            download=False,
            transform=v2.Compose([transform,simple_transform]),
            target_transform=None	
            )

        test_data = datasets.FashionMNIST(
            root=dataset_dir,	# directory to download the data
            train=False, 	# train or test data
            download=False,
            transform=simple_transform, # no need to transform test data
            target_transform=None
            )

        train_dataloader = DataLoader(dataset=train_data, 
                                    batch_size=BATCH_SIZE,
                                    shuffle=True
                                    )
        test_dataloader = DataLoader(dataset=test_data, 
                                    batch_size=BATCH_SIZE,
                                    shuffle=False
                                    ) 
        return train_dataloader,test_dataloader
    except RuntimeError: #RuntimeError when PyTorch cannot find the dataset
        print(f"\n The dataset '{dataset_name}' does not correspond to "
              "any dataset.\n")
        sys.exit(1)


if __name__ == "__main__": 
    train,test = create_train_test_dataloaders("some spaces are good") 
    print(len(train))