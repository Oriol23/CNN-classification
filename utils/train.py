
from data_loaders import create_train_test_dataloaders

train,test = create_train_test_dataloaders()

print(len(train),len(test))