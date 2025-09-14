"""Contains model architectures for image classification as torch.nn.Module."""

import torch
from torch import nn 


class TinyVGG_1(nn.Module):
    """Creates a smaller version of the VGG-Net architecture.

    (No padding on the convolution)
    Input       input channels x 28 x 28 
    #Layer 1    
        Conv    hidden_channels x 26 x 26
        Conv    hidden_channels x 24 x 24
        Pool    hidden_channels x 12 x 12
    #Layer 2
        Conv    hidden_channels x 10 x 10
        Conv    hidden_channels x 8 x 8
        Pool    hidden_channels x 4 x 4
    #Layer 3
        FC (fully-connected)    hidden_channels*4*4 -> output_shape

    Args:
        input_shape: An integer indicating the number of channels of the input.
        hidden_channels: An integer indicating the number of channels between 
            layers.
        output_shape: An integer indicating the number of output units.
            Equals number of classes. 
    """
    def __init__(self, input_shape: int, hidden_channels: int, output_shape: int) -> None:
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                    out_channels=hidden_channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 
                      hidden_channels, 
                      kernel_size=3, 
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(hidden_channels,
                      hidden_channels, 
                      kernel_size=3, 
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 
                      hidden_channels, 
                      kernel_size=3, 
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.layer_3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_channels*4*4, 
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        #x = self.layer_1(x)
        #x = self.layer_2(x)
        #x = self.layer_3(x)
        #return x
        return self.layer_3(self.layer_2(self.layer_1(x))) 


#Possible changes
        #Changing number of epochs and number of hidden channels
    #TinyVGG_1 with padding 1
    #TinyVGG_2 with doubling channels every pool
    #TinyVGG_3.0 and TinyVGG_3.1 with 2 linear layers or 3 since some models have 3 (LeNet, AlexNet)
        #Changing number of nodes in the hidden layer or layers
    #TinyVGG_3.0 and 3.1 with linear layers with dropout
    #TinyVGG_4 changing around the conv layers, i.e instead of doing 
    #   conv conv pool conv conv pool, try
    #   conv pool conc conv conv pool or smth



#TO DO 
    #Experiment with the whole dataloader in GPU if possible
    #train -> trains a model from the commandline, although I have to figure out how to set hyperparameters, 
        # model architecture and other things in the input
    #inference -> uses a trained model to make a prediction on data
    #also make a more detailed log of the experiments, which explains every experiment_name in more detail. 

