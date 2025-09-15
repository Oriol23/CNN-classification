"""Contains functions for miscellaneous tasks and saving models and results. """

import os 
import glob
import torch
import pickle 
import itertools
import numpy as np
import pandas as pd
from typing import Dict
from typing import Tuple
from datetime import datetime
import matplotlib.pyplot as plt 
import matplotlib.lines as mlines
from torch.utils.tensorboard import SummaryWriter

from utils.config import RUNS_DIRECTORY
from utils.config import RESULTS_FILENAME 
from utils.config import MODELS_DIRECTORY
from utils.config import RESULTS_DIRECTORY
from utils.config import METADATA_FILENAME


def create_writer(experiment_name: str,
                  model_name: str, 
                  extra:str):
    """Creates a SummaryWriter() saving to a specific directory log_dir.

    log_dir is a combination of 
    experiment_logs/runs/experiment_name/model_name/extra.

    Args:        
        experiment_name: Name of the experiment.
        model_name: Name of the model used on the experiment.
        extra: Anything extra to add to the directory, like hyperparameters. 

    Returns:
        A SummaryWriter() instance, saving to log_dir.
    """

    log_dir = os.path.join(RUNS_DIRECTORY, experiment_name, model_name,extra)

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
                                hyperparameters_combination: Dict,
                                ):
    """Creates and saves a dictionary containig information about an experiment. 
    
    When performing an experiment, store a dictionary containing additional 
    information about the experiment, such as the time of the experiment, 
    model used, number of epochs, changes to the model... in another directory, 
    dict_dir.

    dict_dir follows the same naming convention for folders as the 
    SummaryWriter() log_dir. 
    dict_dir is a combination of 
    experiment_logs/results/experiment_name/model_name/extra
    log_dir is a combination of 
    experiment_logs/runs/experiment_name/model_name/extra

    Args:
        experiment_name: Name of the experiment.
        model_name: Name of the model used on the experiment.
        extra: Anything extra to add to the directory, like hyperparameters. 
        train_dataloader: PyTorch dataloader used to train the model.
        test_dataloader: PyTorch dataloader used to test the model.
        model: PyTorch model used.
        optimizer: PyTorch optimizer used.
        loss_fn: PyTorch loss function used.      
        epochs: Number of epochs the model has been trained for.
        hyperparameters_combination: A dictionary containing the name and value
            of every hyperparameter used in the experiment.
    """

    experiment_data = {}

    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M") #dd-mm-yyyy hour:min
    dataset_name = os.path.split(train_dataloader.dataset.root)[-1]#type: ignore
    trn_dataset_size = len(train_dataloader.dataset)             #type: ignore
    tst_dataset_size = len(test_dataloader.dataset)              #type: ignore
    batch_size = train_dataloader.__getstate__()['batch_size']   #type: ignore
    optimizer_name = type(optimizer).__name__
    loss_fn_name = type(loss_fn).__name__


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
    #Stores the hyperparameters
    experiment_data.update({"Hyperparams": hyperparameters_combination})
    
    try: 
        experiment_data.update({"optimizer params": 
                                optimizer.__getstate__()['defaults']})
    except KeyError:
        pass
    experiment_data.update({"model params": model.__getstate__()['_modules']})

    #Storing the data as a pickle file
    dict_dir = os.path.join(RESULTS_DIRECTORY,experiment_name,model_name,extra)
    
    file_path = os.path.join(dict_dir,METADATA_FILENAME)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(experiment_data, f)   
    
    print(f"[INFO] Created experiment metadata, saving to: {dict_dir}...")


def retrieve_metadata(experiment_name:str,model_name:str,extra:str):
    """Loads a dictionary containing information about an experiment.
    
    Loads a dictionary saved as a .pkl file containing extra information on an 
    experiment that has been saved with the function 
    create_and_save_experiment_metadata(). 
    See its documentation for more information on the file path structure used. 
    
    Args:         
        experiment_name: Name of the experiment folder.
        model_name: Name of the model folder.
        extra: Name of the extra folder. 
    Returns: 
        A dictionary. 
    """
    
    metadata_path = os.path.join(RESULTS_DIRECTORY,experiment_name,model_name, 
                                 extra, METADATA_FILENAME)

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
        overwrite: A boolean indicating whether or not an existing model should 
            be overwritten. Defaults to False.
    
    Example usage:
        save_model(model=model_0,
                    model_name="05_going_modular_tingvgg_model.pth")
    """

    #create models directory if it doesn't exist
    os.makedirs(os.path.dirname(MODELS_DIRECTORY), exist_ok=True)

    # model_name should end with '.pt' or '.pth'"
    if not (model_name.endswith(".pth") or model_name.endswith(".pt")):
        print("model_name should end with '.pt' or '.pth', not saving.")
        return
    
    # Create model save path    
    model_save_path = os.path.join(MODELS_DIRECTORY,model_name)

    #already exists and no overwrite -> not saving
    if os.path.exists(model_save_path) and not overwrite: 
        print(f"[INFO] A model already exists at: {model_save_path}, "
              "not overwriting.")
    #already exists and and overwrite -> overwrites
    elif os.path.exists(model_save_path) and overwrite: 
        print(f"[INFO] Overwriting model {model_name} at {model_save_path}")
    #doesn't exist -> saves
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
        A list of dictionaries of all combinations of hyperparameters, e.g 
        [{'Hidden Channels': 20, 'Epochs': 10, 'lr': 0.1},
        {'Hidden Channels': 20, 'Epochs': 10, 'lr': 0.01},
        ...,
        {'Hidden Channels': 100, 'Epochs': 30, 'lr': 0.001}]
    """

    hyperparams_combinations = []
    dict_comb = []

    for hplist in hyperparams.values():
        hyperparams_combinations=combine_lists(hyperparams_combinations,hplist)
    
    for comb in hyperparams_combinations:
        dict_comb.append({key:val for key,val in zip(hyperparams.keys(),comb)})
    
    return dict_comb


def create_dataframe(results:Dict,hyperparameters_combination):
    """Creates a pandas dataframe with the results of an experiment.
    
    Args: 
        results: A dictionary with the accuracy and loss results.
            Obtained with the train function.
        hyperparameters_combination: A dictionary containing the name and value
            of every hyperparameter used in the experiment.
    
    Returns:
        A dataframe containing the values of the hyperparameters and the 
        accuracy and loss for every epoch of an experiment. 
    """
    dict_to_df = {}
    #Accesses the first column and sees the number of datapoints
    n_datapoints = len(results[next(iter(results.keys()))])
    #Stores the hyperparameters
    for key,val in hyperparameters_combination.items():
        hyperparameters_combination.update({key:[val]*n_datapoints})

    #Stores the results
    dict_to_df.update(results)

    resdf = pd.DataFrame(dict_to_df)

    return resdf

def save_dataframe(df:pd.DataFrame,experiment_name:str,
                   model_name:str,extra:str):
    """Saves a dataframe in the Feather format. 
    
    The path to save it is 
    experiment_logs/results/experiment_name/model_name/extra

    Which is the same as the metadata dictionary. Additionally creates columns
    to store experiment_name, model_name and extra.
    
    Args: 
        df: A pandas dataframe to be saved.
        experiment_name: Name of the experiment.
        model_name: Name of the model used on the experiment.
        extra: Anything extra to add to the directory, like hyperparameters. 
    """

    df["Experiment_name"] = [experiment_name]*len(df)
    df["Model_name"] = [model_name]*len(df)
    df["Extra"] = [extra]*len(df)

    dataframe_path = os.path.join(RESULTS_DIRECTORY,experiment_name,
                                      model_name,extra,RESULTS_FILENAME)

    os.makedirs(os.path.dirname(dataframe_path), exist_ok=True)
    df.to_feather(dataframe_path)
    print(f"[INFO] Saving the above results to: {dataframe_path}")


def retrieve_results(experiment_name=None,model_name=None,extra=None):
    """Loads dataframes containing the results of an experiment.
    
    Loads multiple dataframes saved as .feather files containing the results 
    of experiments that have been saved with the function save_dataframe(). 
    See its documentation for more information on the file path structure used. 
    If any argument is not specified then it will use all possibilities for 
    that argument or combination of them. For example: 
    retrieve_results(experiment_name=Experiment_1) will retrieve all results 
    inside the Experiment_1 folder regardless of model_name and extra. 
    retrieve_results() will retrieve all results. 
    Adds a column named ID that stores the number of each dataframe and orders
    the columns so only the hyperparameters are placed before the train_loss 
    column.
    
    Args:         
        experiment_name: Name of the experiment folder.
        model_name: Name of the model folder.
        extra: Name of the extra folder. 
    
    Returns: 
        A dataframe. 
    """

    if experiment_name is None:
        experiment_name = "*"
    if model_name is None:
        model_name = "*"
    if extra is None:
        extra = "*"

    print(f"[INFO] retrieving results from {RESULTS_DIRECTORY}")
    list_of_paths = glob.glob(os.path.join(RESULTS_DIRECTORY,experiment_name,
                                           model_name,extra,RESULTS_FILENAME))
    if list_of_paths == []:
        print(f"[WARNING] The provided folder names '{experiment_name}','{model_name}','{extra}' "
              "or its combination are not valid.")
        return
    
    results = pd.DataFrame()
    for n,dataframe_path in enumerate(list_of_paths):
        df = pd.read_feather(dataframe_path)
        # Create a unique identifier for each dataframe every time you load them
        df["ID"] = [n]*len(df)
        
        # The columns will be reordered so hyperparameters are always first, 
        # followed by train_loss and the rest does not really matter

        # The columns before training_loss are always hyperparameters if the dfs
        # have not been merged yet
        new_col = df.columns.to_list() # New columns
        curr_col = results.columns.to_list() # Current columns
        # Ordered union of curr_col and new_col
        all_col = curr_col[:]
        [all_col.append(el) for el in new_col if el not in curr_col] 
        # Find the hyperparameter columns of current and new
        new_hp = new_col[0:new_col.index('train_loss')]
        if 'train_loss' in curr_col: # In the first iteration results is empty
            current_hp = curr_col[0:curr_col.index('train_loss')]
        else:
            current_hp = []
        # Ordered union of current_hp and new_hp
        curr_hp_ordered = current_hp[:]
        [curr_hp_ordered.append(el) for el in new_hp if el not in current_hp] 
        # We add the rest of possible new columns
        all_columns_ordered = curr_hp_ordered[:]
        [all_columns_ordered.append(el) for el in all_col if el not in curr_hp_ordered]
        # Concatenate the dataframes
        results = pd.concat([results,df],ignore_index=True)
        # Reorder the columns
        results = results[all_columns_ordered]

    return results