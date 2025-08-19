import torch
from train import train
from model_architectures import TinyVGG
from data_loaders import create_train_test_dataloaders
from config import device 
from torch import nn
from utils.helpers import create_writer

train_dataloader, test_dataloader = create_train_test_dataloaders()

model_0 = TinyVGG(input_shape=1,hidden_channels=10,output_shape=10).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.01)
print(f"Is the model on GPU{next(model_0.parameters()).is_cuda}")
writer = create_writer(experiment_name="Test_writer",model_name="TinyVGG",extra="1 epochs")
train(model=model_0,train_dataloader=train_dataloader,test_dataloader=test_dataloader,optimizer=optimizer,loss_fn=loss_fn,epochs=1,writer=writer)