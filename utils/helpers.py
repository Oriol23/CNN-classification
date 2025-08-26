"""Contains functions for miscellaneous tasks and saving models and results. """
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
from typing import Dict
import pandas as pd

from utils.config import RUNS_DIRECTORY
from utils.config import METADATA_FILENAME
from utils.config import MODELS_DIRECTORY
from utils.config import RESULTS_DIRECTORY
from utils.config import RESULTS_FILENAME

def create_writer(experiment_name: str,
                  model_name: str, 
                  extra=None):
    """Creates a SummaryWriter() saving to a specific directory log_dir.

    log_dir is a combination of 
    experiment_logs/runs/experiment_name/model_name/extra.

    Args:        
        experiment_name: Name of the experiment.
        model_name: Name of the model used on the experiment.
        extra (optional): Anything extra to add to the directory. 
            Defaults to None.

    Returns:
        A SummaryWriter() instance, saving to log_dir.
    """

    if extra:
        log_dir = os.path.join(RUNS_DIRECTORY, experiment_name, model_name, 
                               extra)
    else:
        log_dir = os.path.join(RUNS_DIRECTORY, experiment_name, model_name)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)


def create_and_save_experiment_metadata(experiment_name: str,
                                model_name: str, 
                                extra:str,
                                train_dataloader: torch.utils.data.DataLoader,
                                test_dataloader: torch.utils.data.DataLoader,
                                model: torch.nn.Module,
                                optimizer: torch.optim.Optimizer,
                                loss_fn: torch.nn.Module,
                                epochs: int,
                                ):
    """Creates and saves a dictionary containig information about an experiment. 
    
    When performing an experiment and logging its results on tensorboard, also 
    store a dictionary containing additional information about the experiment, 
    such as the time of the experiment, model used, number of epochs, changes 
    to the model... in another directory, dict_dir.

    dict_dir follows the same naming convention for folders as the 
    SummaryWriter() log_dir. 
    dict_dir is a combination of 
    experiment_logs/results/experiment_name/model_name/extra
    log_dir is a combination of 
    experiment_logs/runs/experiment_name/model_name/extra

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
    #relpath = os.path.relpath(writer.__getstate__()['log_dir'],
    #                                           RUNS_DIRECTORY)
    #[model_name, experiment_name, extras (if present)]
    #components = relpath.split(os.sep) 
    
    #model_name = components[0]
    #experiment_name = components[1]

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
    if extra:
        dict_dir = os.path.join(RESULTS_DIRECTORY,experiment_name,model_name,
                                extra)
    else:
        dict_dir = os.path.join(RESULTS_DIRECTORY,experiment_name,model_name)
    
    file_path = os.path.join(dict_dir,METADATA_FILENAME)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(experiment_data, f)   
    
    print(f"[INFO] Created experiment metadata, saving to: {dict_dir}...")


def retrieve_metadata(experiment_name:str,model_name:str,extra=None):
    """Loads a pickle file containing information about an experiment.
    
    Loads a dictionary containing extra information on an experiment that has
    been saved with the function create_and_save_experiment_metadata(). 
    See its documentation for more information on the file path structure used. 
    
    Args:         
        experiment_name: Name of the experiment.
        model_name: Name of the model used.
        extra (optional): Anything extra to add. Defaults to None.
    
    Returns: 
        A dictionary. 
    """
    
    if extra:
        metadata_path = os.path.join(RESULTS_DIRECTORY, experiment_name,
                                     model_name, extra, METADATA_FILENAME)
    else:
        metadata_path = os.path.join(RESULTS_DIRECTORY, experiment_name, 
                                     model_name, METADATA_FILENAME)
    
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
    

def save_model(model: torch.nn.Module,
                model_name: str,
                overwrite=False):
    """Saves a PyTorch model to the project's models directory.

    Args:
        model: A target PyTorch model to save.
        model_name: A filename for the saved model. Should include
            either ".pth" or ".pt" as the file extension.
        overwrite: A boolean indicating whether or not an existing file is 
            overwritten. Defaults to False.
    
    Example usage:
        save_model(model=model_0,
                    model_name="05_going_modular_tingvgg_model.pth")
    """
    #create models directory if it doesn't exist
    os.makedirs(os.path.dirname(MODELS_DIRECTORY), exist_ok=True)

    #assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    if not (model_name.endswith(".pth") or model_name.endswith(".pt")):
        print("model_name should end with '.pt' or '.pth', not saving.")
        return
    
    # Create model save path    
    model_save_path = os.path.join(MODELS_DIRECTORY,model_name)
    
    if os.path.exists(model_save_path) and not overwrite: #exists no overwrite -> not saving
        print(f"[INFO] A model already exists at: {model_save_path}, not overwriting.")
    elif os.path.exists(model_save_path) and overwrite: #exists and overwrite -> overwrites
        print(f"[INFO] Overwriting model {model_name} at {model_save_path}")
    else:
        # Save the model state_dict()
        torch.save(obj=model.state_dict(),
                    f=model_save_path)    
        print(f"[INFO] Saving model to: {model_save_path}")


def flatten(mixed_list):
    """Flattens a list that contains numbers and other lists.
    
    Args: 
        mixed_list: A list containing numbers and lists.
    
    Returns: 
        A list of all the elements.
    """
    flattened_list = []
    for elem in mixed_list:
        if isinstance(elem,list):
            for el in elem:
                flattened_list.append(el)
        else:
            flattened_list.append(elem)
    return flattened_list
    

def combine_lists(list_1, list_2):
    """Returns a list of all pairs of elements of list 1 and 2.
    
    If any of the two lists is empty returns the other. 
    If any of the elements of the list is another list, it flattens it. 

    Args: 
        list_1: List number one.
        list_2: List number two.
    
    Returns: 
        A list of lists. 
    """
    if list_1 == []:
        return list_2
    if list_2 == []:
        return list_1
    
    list_combinations = []
    for el_1 in list_1:
        for el_2 in list_2:
            if isinstance(el_1, list) or isinstance(el_2, list):
                list_combinations.append(flatten([el_1,el_2]))
            else:
                list_combinations.append([el_1,el_2])
    return list_combinations


def hyperparameter_combinations(hyperparams: Dict[str,list]):
    """Returns all possible combinations of hyperparameters.
    
    Args: 
        hyperparams: A dictionary containing the hyperparameters.
            The key must be a string and the value a list.  
    
    For example: 
        HYPERPARAMETERS = {'Hidden Channels': [20,40,60,80,100],
                            'Epochs': [10,20,30],
                            'lr': [0.1,0.01,0.001] } 
                            
    Returns: 
        A list of tuples of all possible combinations, e.g 
        [(20,10,0.1),(20,10,0.01),...,(100,30,0.001)]
    """

    hyperparams_list = []
    hyperparams_combinations = []
    tuple_combinations = []

    for key in hyperparams.keys():
        hyperparams_list.append(hyperparams[key])
    
    for hplist in hyperparams_list:
        hyperparams_combinations = combine_lists(hyperparams_combinations,hplist)

    for combination in hyperparams_combinations:
        tuple_combinations.append(tuple(combination))
    
    return tuple_combinations


def create_dataframe(results,hyperparameters_tuple,hyperparameters_keys):
    """Creates a pandas dataframe with the results of an experiment.
    
    Args: 
        results: A dictionary with the accuracy and loss results.
            Obtained with the train function.
        hyperparameters_tuple: A tuple with the values of the hyperparameters 
            used in the experiment.
        hyperparameters_keys: The keys of the hyperparameters dictionary used.
    
    Returns:
        A dataframe containing the values of the hyperparameters and the 
        accuracy and loss for every epoch of an experiment. 
    """
    dict_to_df = {}
    #Accesses the first value and sees the number of datapoints
    n_datapoints = len(results[next(iter(results.keys()))])
    #Stores the hyperparameters
    for m,key in enumerate(hyperparameters_keys):
        dict_to_df.update({key:[hyperparameters_tuple[m]]*n_datapoints})
    #Stores the results
    dict_to_df.update(results) 

    resdf = pd.DataFrame(dict_to_df)

    return resdf

def save_dataframe(df,experiment_name,model_name,extra=None):
    """Saves a dataframe in the Feather format. 
    
    The path to save it is 
    experiment_logs/results/experiment_name/model_name/extra

    Which is the same as the metadata dictionary.
    
    Args: 
        df: A pandas dataframe to be saved.        
        experiment_name: Name of the experiment
        model_name: Name of the model used
        extra: Anything extra to add.
    """

    if extra:
        dataframe_path = os.path.join(RESULTS_DIRECTORY,experiment_name,
                                      model_name,extra,RESULTS_FILENAME)
    else: 
        dataframe_path = os.path.join(RESULTS_DIRECTORY,experiment_name,
                                      model_name,RESULTS_FILENAME)
    
    os.makedirs(os.path.dirname(dataframe_path), exist_ok=True)
    df.to_feather(dataframe_path)
    print(f"[INFO] Saving the above results to: {dataframe_path}")


def retrieve_results(experiment_name:str,model_name:str,extra=None) -> pd.DataFrame:
    """Loads a feather file containing the results of an experiment.
    
    Loads a dataframe containing the results of an experiment that has
    been saved with the function save_dataframe(). See its documentation
    for more information on the file path structure used. 
    
    Args:         
        experiment_name: Name of the experiment.
        model_name: Name of the model used.
        extra (optional): Anything extra to add. Defaults to None.
    
    Returns: 
        A dataframe. 
    """
    
    if extra:
        dataframe_path = os.path.join(RESULTS_DIRECTORY, experiment_name,
                                     model_name, extra, RESULTS_FILENAME)
    else:
        dataframe_path = os.path.join(RESULTS_DIRECTORY, experiment_name, 
                                     model_name, RESULTS_FILENAME)
    
    if os.path.exists(dataframe_path):
        results = pd.read_feather(dataframe_path)
        return results
    else:
        print(f"No results stored in {dataframe_path}")
        return pd.DataFrame()
