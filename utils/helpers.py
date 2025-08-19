# The .py files in the utils folders will be

#get data -> downloads a dataset (skip for this project)
#dataset -> creates datasets & dataloaders from the raw data
#model_architectures -> classes with the model architectures
#engine -> functions for training and testing models
            #make a monitored training function that tests loss and acc every epoch and a barebones training
            #in case tensorboard automatically tracks that, or for the train script. 
#helpers -> saving and loading a model, creating writers
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


old_code = """ def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra=None):
    "Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

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
    "
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
    return SummaryWriter(log_dir=log_dir) """


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
        A tuple with a torch.utils.tensorboard.writer.SummaryWriter(), saving to log_dir, and the directory to
        save a dictionary with more information on the experiment.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    import os

    if extra:
        log_dir = os.path.join(MAIN_DIRECTORY, "runs", model_name, experiment_name, extra)
        dict_dir = os.path.join(MAIN_DIRECTORY, "runs_metadata", model_name, experiment_name, extra)
    else:
        log_dir = os.path.join(MAIN_DIRECTORY, "runs", model_name, experiment_name)
        dict_dir = os.path.join(MAIN_DIRECTORY, "runs_metadata", model_name, experiment_name)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir), dict_dir
    
def create_experiment_metadata(experiment_name: str, 
                               train_dataloader: torch.utils.data.DataLoader,
                               test_dataloader: torch.utils.data.DataLoader,
                               model: torch.nn.Module,
                               optimizer: torch.optim.Optimizer,
                               loss_fn: torch.nn.Module,
                               epochs: int,
                               dict_dir: str,
                               #others: dict
                               ):
    """Creates and stores a dictionary containig all the information about an experiment. 
    
    When performing an experiment and logging its results on tensorboard, also store a dictionary containing
    information about the experiment, such as the time, model used, number of epochs, changes to the model...

    Args:
        experiment_name: name of the experiment to log
        train_dataloader: PyTorch dataloader used to train the model in the experiment
        test_dataloader: PyTorch dataloader used to test the model in the experiment
        model: PyTorch model used
        optimizer: PyTorch optimizer used
        loss_fn: PyTorch loss function used
        epochs: number of epochs the model has been trained on
        others: other hyperparameters of the model, as a dictionary. 
            For example: {"Conv layers hidden nodes": 120}
        dict_dir: directory where the dictionary will be stored. Mirrors 

    """
    from datetime import datetime
    import os
    import pickle 

    experiment_data = {}

    #experiment_name
    timestamp = datetime.now().strftime("%d-%m-%Y")
    dataset_name = train_dataloader.dataset.root.split("\\")[-1]    #type: ignore
    trn_dataset_size = len(train_dataloader.dataset)                    #type: ignore
    tst_dataset_size = len(test_dataloader.dataset)                     #type: ignore
    batch_size = train_dataloader.__getstate__()['batch_size']          #type: ignore
    model_name = model._get_name()
    optimizer_name = type(optimizer).__name__
    loss_fn_name = type(loss_fn).__name__
    #epochs
    #other hyperparameters like hidden channels which will be tricky to implement smoothly into the dict, since
    #I would need to provide the name and value of the hyperparameter, i.e {conv initial hidden channels: 64} or 
    # {FC hidden layer nodes: 120}, although I might be able to use the experiment_name

    # OPTIMIZER PARAMETERS LIKE LEARNING RATE, MOMENTUM AND ON AND ON


    experiment_data.update({"experiment_name": experiment_name})
    experiment_data.update({"date": timestamp})
    experiment_data.update({"model": model_name})
    experiment_data.update({"dataset": dataset_name})
    experiment_data.update({"training dataset size": trn_dataset_size})
    experiment_data.update({"testing dataset size": tst_dataset_size})
    experiment_data.update({"batch size": batch_size})
    experiment_data.update({"optimizer": optimizer_name})
    experiment_data.update({"loss function": loss_fn_name})
    experiment_data.update({"epochs": epochs})
    #experiment_data.update(others)

    filename = "Metadata.pkl"
    file_path = os.path.join(dict_dir,filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(experiment_data, f)