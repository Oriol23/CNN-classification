"""Contains functions for training and testing a PyTorch model."""
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
#from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_step(model: torch.nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                loss_fn: torch.nn.Module, 
                optimizer: torch.optim.Optimizer,
                device=device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to calculate loss.
        optimizer: A PyTorch optimizer to minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:

        (0.1112, 0.8743)
    """
    model.train()
    train_loss, train_acc = 0, 0
    for X,y in dataloader:
        # Put data to device
        X, y = X.to(device), y.to(device)
        # Make predictions
        y_pred = model(X)
        # Calculate loss
        loss = loss_fn(y_pred,y)
        train_loss += loss.item()
        # Zero gradient
        optimizer.zero_grad()
        # Calculate loss gradient
        loss.backward()
        # Optimizer step
        optimizer.step()
        # Calculate accuracy 
        train_acc += (y==y_pred.argmax(dim=1)).sum().item()/len(y)        
    # Average accuracy
    train_acc /= len(dataloader)
    train_loss /= len(dataloader) 

    #print(f"Train loss: {train_loss:.4f} | Train acc: {100*train_acc:.0f}%")
    return train_loss, train_acc


def test_step(model: torch.nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                loss_fn: torch.nn.Module,
                device=device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:

        (0.0223, 0.8985)
    """	
    model.eval()
    test_loss, test_acc = 0,0
    with torch.inference_mode():
        for X,y in dataloader:
            # Put data to device
            X, y = X.to(device), y.to(device)
            # Make predictions
            y_pred = model(X)
            # Accumulate loss and accuracy per batch
            loss = loss_fn(y_pred,y)
            test_loss += loss.item()
            test_acc += (y==y_pred.argmax(dim=1)).sum().item()/len(y) 
                    #y is (batch_size) and y_pred is (batch_size,n_classes)
        test_acc /= len(dataloader)
        test_loss /= len(dataloader)


    #print(f"Test loss: {test_loss:.4f} | Test acc: {100*test_acc:.0f}%")
    return test_loss, test_acc


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int, 
          writer=None,  
          device=device
          ):
    """Trains and tests a PyTorch model.

    Passes a target PyTorch model through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates accuracy and loss for training and testing datasets.

    Stores metrics to specified writer log_dir if present.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        writer: A SummaryWriter() instance to log model results to.
        device: A target device to compute on (e.g. "cuda" or "cpu").
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": [],
               "Epoch #": []
    }

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["Epoch #"].append(epoch+1)

        ### Use the writer parameter to track experiments ###
        # See if there's a writer, if so, log to it
        if writer:
            # Add results to SummaryWriter
            writer.add_scalars(main_tag="Loss", 
                               tag_scalar_dict={"train_loss": train_loss,
                                                "test_loss": test_loss},
                               global_step=epoch)
            writer.add_scalars(main_tag="Accuracy", 
                               tag_scalar_dict={"train_acc": train_acc,
                                                "test_acc": test_acc}, 
                               global_step=epoch)

            # Close the writer
            writer.flush()
            writer.close()
        else:
            pass
    # Return the filled results at the end of the epochs
    return results