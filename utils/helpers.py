# The .py files in the utils folders will be

#get data -> downloads a dataset
#data_loaders -> creates dataloaders from a dataset
#model_architectures -> classes with the model architectures
#engine -> functions for training and testing models
            #make a monitored training function that tests loss and acc every epoch and a barebones training
            #in case tensorboard automatically tracks that, or for the train script. 
#helpers -> saving and loading a model, creating writers, storing metadata
#config -> stores variables like the paths to the main project folder and the data folder
#
#plots -> plotting results, probably better in a notebook so adjust, might not be needed because of tensorboard
#train -> trains a model from the commandline, although I have to figure out how to set hyperparameters, 
    # model architecture and other things in the input
#inference -> uses a trained model to make a prediction on data


#next(model.parameters()).is_cuda #check if model on cuda

#tensorboard --logdir=experiment_logs/runs      #To open tensorboard click the link

#Pytorch tensor format is (batch,C,H,W)


import os 
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import pickle 

from config import RUNS_DIRECTORY
from config import RUNS_METADATA_DIRECTORY
from utils.config import METADATA_FILENAME


def create_writer(model_name: str,
                  experiment_name: str, 
                  extra=None):
    """Creates a SummaryWriter() saving to a specific directory log_dir.

    log_dir is a combination of /runs/model_name/experiment_name/extra.

    Args:
        model_name: Name of the model used on the experiment.
        experiment_name: Name of the experiment.
        extra (optional): Anything extra to add to the directory. 
            Defaults to None.

    Returns:
        A SummaryWriter() instance, saving to log_dir.
    """

    if extra:
        log_dir = os.path.join(RUNS_DIRECTORY, model_name, 
                               experiment_name, extra)
    else:
        log_dir = os.path.join(RUNS_DIRECTORY, model_name, 
                               experiment_name)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)

        

def create_experiment_metadata(writer, #tensorboard SummaryWriter()
                               train_dataloader: torch.utils.data.DataLoader,
                               test_dataloader: torch.utils.data.DataLoader,
                               model: torch.nn.Module,
                               optimizer: torch.optim.Optimizer,
                               loss_fn: torch.nn.Module,
                               epochs: int,
                               ):
    """Stores a dictionary containig more information about an experiment. 
    
    When performing an experiment and logging its results on tensorboard, also 
    store a dictionary containing additional information about the experiment, 
    such as the time of the experiment, model used, number of epochs, changes 
    to the model... in another directory, dict_dir.

    dict_dir follows the same naming convention for folders as the 
    SummaryWriter() log_dir. 
    dict_dir is a combination of /runs_metadata/model_name/experiment_name/extra
    log_dir is a combination of /runs/model_name/experiment_name/extra

    Args:
        writer: A SummaryWriter() instance.
        train_dataloader: PyTorch dataloader used to train the model.
        test_dataloader: PyTorch dataloader used to test the model.
        model: PyTorch model used.
        optimizer: PyTorch optimizer used.
        loss_fn: PyTorch loss function used.      
        epochs: Number of epochs the model has been trained for.
    """

    experiment_data = {}

    #relative path of the SummaryWriter() logdir and the runs folder, containing
    #model_name/experiment_name/extras 
    relpath = os.path.relpath(writer.__getstate__()['log_dir'],
                                               RUNS_DIRECTORY)
    #[model_name, experiment_name, extras (if present)]
    components = relpath.split(os.sep) 
    
    model_name = components[0]
    experiment_name = components[1]
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M") #dd-mm-yyyy hour:min
    dataset_name = os.path.split(train_dataloader.dataset.root)[-1]#type: ignore
    trn_dataset_size = len(train_dataloader.dataset)             #type: ignore
    tst_dataset_size = len(test_dataloader.dataset)              #type: ignore
    batch_size = train_dataloader.__getstate__()['batch_size']   #type: ignore
    optimizer_name = type(optimizer).__name__
    loss_fn_name = type(loss_fn).__name__

    #other hyperparameters like hidden channels which will be tricky to 
    #implement smoothly into the dict, since I would need to provide the name 
    #and value of the hyperparameter, i.e {conv initial hidden channels: 64} or 
    #{FC hidden layer nodes: 120}

    experiment_data.update({"date": timestamp})
    experiment_data.update({"experiment name": experiment_name})
    experiment_data.update({"model name": model_name})
    #if the optimizer has learning rate
    try: 
        learning_rate = optimizer.__getstate__()['defaults']['lr']
        experiment_data.update({"learning rate": learning_rate})
    except KeyError:
        pass
    experiment_data.update({"epochs": epochs})
    experiment_data.update({"loss function": loss_fn_name})    
    experiment_data.update({"optimizer name": optimizer_name})
    experiment_data.update({"dataset": dataset_name})
    experiment_data.update({"training dataset size": trn_dataset_size})
    experiment_data.update({"testing dataset size": tst_dataset_size})
    experiment_data.update({"batch size": batch_size})
    try: 
        experiment_data.update({"optimizer params": 
                                optimizer.__getstate__()['defaults']})
    except KeyError:
        pass

    experiment_data.update({"model params": model.__getstate__()['_modules']})
    #experiment_data.update(others)

    #Storing the data as a pickle file
    dict_dir = os.path.join(RUNS_METADATA_DIRECTORY,relpath)

    file_path = os.path.join(dict_dir,METADATA_FILENAME)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(experiment_data, f)   
    
    print(f"[INFO] Created experiment metadata, saving to: {dict_dir}...")




def retrieve_metadata(model_name:str, experiment_name:str, extra=None):
    """Loads a pickle object containing information about an experiment.
    
    Loads a dictionary containing extra information on an experiment that has
    been saved with the function create_experiment_metadata(). See documentation
    for more information on the file path structure used. 
    
    Args: 
        model_name: Name of the model used.
        experiment_name: Name of the experiment.
        extra (optional): Anything extra to add. Defaults to None.
    
    Returns: 
        A dictionary. 
    """
    
    if extra:
        metadata_path = os.path.join(RUNS_METADATA_DIRECTORY, model_name, 
                               experiment_name, extra, METADATA_FILENAME)
    else:
        metadata_path = os.path.join(RUNS_METADATA_DIRECTORY, model_name, 
                               experiment_name, METADATA_FILENAME)
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as file:
            metadata = pickle.load(file)
        return metadata
    else:
        print(f"No metadata stored in {metadata_path}")


def dataloader_memory(dataloader):
    """Prints the size in MB of a batch and the whole dataloader.
    
    Args: 
        dataloader: A PyTorch torch.utils.data.DataLoader object.
    """

    batch,label = next(iter(dataloader))
    #size = # elements in a batch * element size in bytes / conversion to MB 
    size = batch.nelement() * batch.element_size() / 1024**2 #Size in MB
    nbat = len(dataloader)

    print(f'The size of a batch is {size:.3f} MB \n'
        f'The size of the train/test dataset is {size*nbat:.3f} MB')
    

