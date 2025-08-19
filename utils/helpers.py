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
import torch
from torch.utils.tensorboard import SummaryWriter
from config import MAIN_DIRECTORY
def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra=None):
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    from datetime import datetime
    import os

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join(MAIN_DIRECTORY, "runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join(MAIN_DIRECTORY, "runs", timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)