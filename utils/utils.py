# The .py files in the utils folders will be

#get data -> downloads a dataset (skip for this project)
#dataset -> creates datasets & dataloaders from the raw data
#model_builder -> classes with the model architectures
#engine -> functions for training and testing models
            #make a monitored training function that tests loss and acc every epoch and a barebones training
            #in case tensorboard automatically tracks that, or for the train script. 
#utils -> saving and loading a model
#
#plots -> plotting results, probably better in a notebook so adjust
#train -> trains a model from the commandline, although I have to figure 
    #out how to set hyperparameters, model architecture and other things 
    #in the input
#inference -> uses a trained model to make a prediction on data


#next(tvgg.parameters()).is_cuda #check if model on cuda
#Pytorch tensor format is (batch,C,H,W)
