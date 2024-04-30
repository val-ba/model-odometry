
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from OdometryDataset import OdomDataset
from SimpleModel import RecurrentNet, MLP, LSTMNet
from os.path import join
import argparse
import yaml
import pathlib
import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


# we create dicts which link strings to classes for arg parser
criterion_dict = {"mse": nn.MSELoss(), "mae": nn.L1Loss()}
model_architecture_dict = {"RNN": RecurrentNet, "MLP": MLP, "LSTM" : LSTMNet}
optimizer_dict = {"adam": optim.Adam, "sgd": optim.SGD}
func_dict = {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_path = join(str(pathlib.Path(__file__).parent.parent.resolve()),"Data/")

RNN_names = ["RNN", "LSTM"]

def initialize_training(architecture, dataset, hidden_sizes: str, dropout: float, criterion: str, optimizer: str, learning_rate: float, activation_function, recurrent_size=None, recurrent_depth=None):
    """creates a model, criterion and optimizer to use in training

    Args:
        architecture (str): architecture name as taken from the parser
        dataset (OdomDataset): dataset which was previously created
        hidden_sizes (str): hidden sizes of the model as a string in the form of "32|64|128"
        dropout (float): dropout rate
        criterion (str): string of the criterion name
        optimizer (str): string of the optimizer name
        learning_rate (float): learning rate for the optimizer
        activation_function (str): string of the activation function name to be used for the dict
        recurrent_size (_type_, optional): size of the recurrent layers. Only applicable for RNN or LSTM architecures. Defaults to None.
        recurrent_depth (_type_, optional): depth of the recurrent layer. Only applicable for RNN or LSTM architectures. Defaults to None.

    Returns:
        model (torch.nn.Module): model to be trained
        criterion (torch.nn.Module): criterion to be used for training
        optimizer (torch.optim.Optimizer): optimizer to be used for training
    """
    hidden_sizes = [int(x) for x in hidden_sizes.split("|")]
    activation_function = func_dict[activation_function]
    if architecture in RNN_names:
        model = model_architecture_dict[architecture](len(dataset.data_names), hidden_sizes, dropout=dropout, func=activation_function, recurrent_size=recurrent_size, recurrent_depth=recurrent_depth)
    else:
        model = model_architecture_dict[architecture](len(dataset.data_names), hidden_sizes, dropout, func=activation_function)
    criterion = criterion_dict[criterion]
    optimizer = optimizer_dict[optimizer](model.parameters(), lr=learning_rate)

    return model, criterion, optimizer

def make_train_and_test_loader(dataset, batch_size=64, train_percentage=0.8, different_val_set=None):
    """creates a train and test loader from a dataset
    Args:
        dataset (OdomDataset): dataset to be split
        batch_size (int, optional): batch size. Defaults to 64.
        train_percentage (float, optional): percentage of the dataset to be used for training. Defaults to 0.8.
        different_val_set (OdomDataset, optional): if a different validation set is to be used. Defaults to None.
    """
    if not different_val_set:
        train_size = int(train_percentage * len(dataset))
        test_size = len(dataset) - train_size
        batch_size = batch_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    else:
        train_dataset = dataset
        test_dataset = different_val_set

    trainloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    return trainloader, testloader

def calc_single_loss(loss, loss_split, architecture):
    """calculates the loss for x, y and yaw and the total loss

    Args:
        loss : total loss
        loss_split : loss split into x, y and yaw
        architecture : architecture name
    Returns:
        x_loss, y_loss, yaw_loss, total_loss
    """
    if architecture == "MLP":
        sample_size = len(loss_split)
        x_loss = loss_split[:,0].sum() / sample_size
        y_loss = loss_split[:,1].sum()/ sample_size
        yaw_loss = loss_split[:,2].sum()/ sample_size

    else:
        sample_size = len(loss_split)*len(loss_split[0])
        x_loss = loss_split[:,:,0].sum() / sample_size
        y_loss = loss_split[:,:,1].sum()/ sample_size
        yaw_loss = loss_split[:,:,2].sum()/ sample_size
    return x_loss.item(), y_loss.item(), yaw_loss.item(), loss.item()

def train_loop(dataloader, model, loss_fn, optimizer, architecture, wandb=None):
    """train loop for the model

    Args:
        dataloader (OdometryDataset) : dataloader to be used
        model (torch.nn.Module): model to be trained
        loss_fn (torch.nn.Module): loss function to be used
        optimizer (torch.optim.Optimizer): optimizer to be used
        architecture (str): architecture name
        wandb (wandb, optional): wandb object to log the data. Defaults to None.
        
    """
    model.to(device)
    try:
        model.train()
        size = len(dataloader.dataset)
        for batch, (y, X) in enumerate(dataloader):
            if batch % 100 == 0:
                print(f"Batch {batch} of {size//dataloader.batch_size}", end="\r")
            X, y = X.to(device), y.to(device)
            if architecture == "RNN":
                init_hidden = model.init_hidden(X.size()[0]) 
                pred, _ = model(X.float(), init_hidden.to(device))
            elif architecture == "LSTM":
                init_hidden, init_cell = model.init_hidden(X.size()[0])
                pred, _ = model(X.float(), (init_hidden.to(device), init_cell.to(device)))
            else:
                pred = model(X.float())
            loss = loss_fn(pred, y.float())
            if batch % 1000 == 0:
                loss_split =nn.L1Loss(reduction="none")(pred, y.float())
                x_loss, y_loss, yaw_loss, total_loss = calc_single_loss(loss, loss_split, architecture) 
                if wandb:
                    wandb.log({"x_loss": x_loss})
                    wandb.log({"y_loss": y_loss})
                    wandb.log({"yaw_loss": yaw_loss})
                    wandb.log({"loss": loss})
                else:
                    print(f"loss: {total_loss}")
            # update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    except Exception as e:
        print(e)
        print(f"Batch {batch} failed. Skipping...")

def test_loop(dataloader, model, loss_fn, architecture, wandb=None, return_loss=False):
    """test loop for the model

    Args:
        dataloader (OdometryDataset): dataloader to be used
        model (torch.nn.Module): model to be tested
        loss_fn (torch.nn.Module): loss function to be used
        architecture (str): architecture name
        wandb (wandb, optional): wandb object to log the data. Defaults to None.
        return_loss (bool, optional): if the loss should be returned. Defaults to False.
        

    Returns:
        float: loss if return_loss is True
    """
    model.eval()
    # get the average loss over the whole test set
    test_loss = 0
    optuna_loss = 0

    with torch.no_grad():
        for y, X in dataloader:
            X, y = X.to(device), y.to(device)
            if architecture == "RNN":
                init_hidden = model.init_hidden(X.size()[0])
                pred, _ = model(X.float(), init_hidden.to(device))
            elif architecture == "LSTM":
                init_hidden, init_cell = model.init_hidden(X.size()[0])
                pred, _ = model(X.float(), (init_hidden.to(device), init_cell.to(device)))
            else:
                pred = model(X.float())
            test_loss += loss_fn(pred, y.float())
            optuna_loss += nn.L1Loss()(pred, y.float())
            
    test_loss /= len(dataloader.dataset)
    optuna_loss /= len(dataloader.dataset)
    
    if wandb:
        wandb.log({"test_loss": test_loss})

    if return_loss:
        # always uses the same loss function to make it comparable for optuna
        return optuna_loss

def save_model(model, name, data_dict):
    """saves the model and the data_dict

    Args:
        model (torch.nn.Module): model to be saved
        name (str): name of the model
        data_dict (dict): data_dict to be saved
    """
    print(f"Saving model as {name}...")
    model.to("cpu")
    torch.save(model.state_dict(), join(join(str(pathlib.Path(__file__).parent.parent.resolve()), "trained_models/", name +".pth")))
    yaml.dump(data_dict, open(join(str(pathlib.Path(__file__).parent.parent.resolve()) , "config/odometry_model_parameters_"+name+".yaml"), "w"))
    model.to(device)  

def test_and_train_loop(model_data_dict, dataloader, testloader, model, criterion, optimizer, architecture, wandb=None, epochs=100, project_name="no_name", data_name="Data"):
    """Combines test and train loop and trains it for a number of epochs. The model is also saved every n epochs.

    Args:
        model_data_dict (dict): model data dict to be saved
        dataloader (OdometryDataset): dataloader to be used
        testloader (OdometryDataset): testloader to be used
        model (torch.nn.Module): model to be trained
        criterion (torch.nn.Module): criterion to be used
        optimizer (torch.optim.Optimizer): optimizer to be used
        architecture (str): architecture name
        wandb (wandb, optional): wandb object to log the data. Defaults to None.
        epochs (int, optional): number of epochs to train. Defaults to 100.
        project_name (str, optional): name of the project. Defaults to "no_name".
        data_name (str, optional): name of the data. Defaults to "Data".
        
    """
    if wandb:
        wandb.init(project=project_name,
        config={
        "architecture": architecture,
        "dataset": data_name,
        "optimizer": optimizer,
        "criterion": criterion,
        })
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    for t in tqdm.tqdm(range(epochs)):
        train_loop(dataloader, model, criterion, optimizer, architecture, wandb=wandb)
        test_loss = test_loop(testloader, model, criterion, architecture, wandb=wandb, return_loss=True)
        scheduler.step(test_loss)
        if t % sn == 0 or t ==  epochs - 1:
            name =args.name+str(t)
            save_model(model, name, model_data_dict)

def make_model_dict(architecture, dataset, hidden_sizes, dropout, func, recurrent_size=None, recurrent_depth=None, ):
    """creates a model dict which contains all the information needed to recreate the model architecture

    Args:
        architecture (str): architecture name
        dataset (OdomDataset): dataset to be used
        hidden_sizes (str): hidden sizes of the model as a string in the form of "32|64|128"
        dropout (float): dropout rate
        func (torch function): activation function
        recurrent_size (int, optional): size of the recurrent layers. Only applicable for RNN or LSTM architecures. Defaults to None.
        recurrent_depth (int, optional): depth of the recurrent layer. Only applicable for RNN or LSTM architectures. Defaults to None.
        

    Returns:
        dict: model dict
    """
    hidden_sizes = [int(x) for x in hidden_sizes.split("|")]
    return {"model_odometry": {
        "ros__params" : {
        "model_info": {
            "architecture": architecture,
            "hidden_sizes": hidden_sizes,
            "dropout": float(dropout),
            "recurrent_size": int(recurrent_size),
            "recurrent_depth": int(recurrent_depth),
            "func": func.__name__,
        },
        "dataset_info": {
            "input_col_names": [str(col) for col in dataset.data_names],
        }
        }
    }}  
         
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--architecture", default="LSTM", type=str, help="architecture name", choices=model_architecture_dict.keys())
    parser.add_argument("-o", "--optimizer", default="adam", type=str, help="optimizer", choices=optimizer_dict.keys())
    parser.add_argument("-c", "--criterion", default="mae", type=str, help="criterion", choices=criterion_dict.keys())
    parser.add_argument("-f", "--func", default="relu", type=str, help="activation function", choices=func_dict.keys())
    parser.add_argument("-e", "--epochs", default=10, type=int, help="number of epochs")
    parser.add_argument("-lr", "--learning_rate", default=0.001, type=float, help="learning rate")
    parser.add_argument("-sn", "--save_number", default=10, type=int, help="save model every n epochs")
    parser.add_argument("-n", "--name", default="default_net", type=str, help="name of the run")
    parser.add_argument("-hd", "--hidden_sizes", default="32", type=str, help="hidden sizes of MLP")
    parser.add_argument("-dr", "--dropout", default=0.05, type=float, help="dropout rate")
    parser.add_argument("-d", "--data", default="rosbags", type=str, help="dataset name")
    parser.add_argument("-rs", "--recurrent_size", default=8, type=int, help="recurrent size")
    parser.add_argument("-rd", "--recurrent_depth", default=3, type=int, help="recurrent depth")
    parser.add_argument("-p", "--project_name", default="no_name", type=str, help="project name")
    parser.add_argument("-b", "--batch_size", default=64, type=int, help="batch size")
    args = parser.parse_args()


    architecture = args.architecture
    optimizer = args.optimizer
    criterion = args.criterion
    nr_epochs = args.epochs
    learning_rate = args.learning_rate
    sn = args.save_number
    dropout = args.dropout
    batch_size = args.batch_size
    path = join(str(pathlib.Path(__file__).parent.parent.resolve()), "Data/",args.data)



    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    if architecture in RNN_names:
        dataset = OdomDataset(path, sequence=5)
    else:
        dataset = OdomDataset(path, sequence=1)
    trainloader, testloader = make_train_and_test_loader(dataset, batch_size=batch_size, train_percentage=0.8)

    model, criterion, optimizer = initialize_training(architecture, dataset, args.hidden_sizes, dropout, criterion, optimizer, learning_rate, args.func, args.recurrent_size, args.recurrent_depth)
    model_data_dict = make_model_dict(architecture, dataset, args.hidden_sizes, dropout, func_dict[args.func], recurrent_size=args.recurrent_size, recurrent_depth=args.recurrent_depth)
    print(f"Train size: {len(trainloader.dataset)}")
    test_and_train_loop(model_data_dict, trainloader, testloader, model, criterion, optimizer, architecture, wandb, epochs=nr_epochs, project_name=args.project_name, data_name=args.data)