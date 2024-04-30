import torch
from os.path import join
from Model.SimpleModel import MLP, RecurrentNet, LSTMNet
import yaml
import pathlib
from typing import Tuple

model_architecture_dict = {"RNN": RecurrentNet, "MLP": MLP, "LSTM" : LSTMNet}
func_dict = {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}

def load_model_from_name_string(model_name: str) -> Tuple[torch.nn.Module, str]:
    """Loads a model from a name string and sends it to CPU. The model architecture is read from a specific yaml.

    Args:
        name (str): Name of the model. The model should be saved in the trained_models folder and the parameters should be saved in the config folder.
    """
    _package_path = str(pathlib.Path(__file__).parent.parent.resolve())
    params = yaml.load(open(join(_package_path, "config/odometry_model_parameters_"+model_name+".yaml"), "r"), Loader=yaml.FullLoader)["model_odometry"]["ros__params"]
    architecture = params["model_info"]["architecture"]
    input_size = len(params["dataset_info"]["input_col_names"])
    hidden_sizes = params["model_info"]["hidden_sizes"]
    dropout = params["model_info"]["dropout"]
    func = func_dict[params["model_info"]["func"]]
    if architecture in ["LSTM", "RNN"]:
        recurrent_depth = params["model_info"]["recurrent_depth"]
        recurrent_size = params["model_info"]["recurrent_size"]
        if architecture == "RNN":
            model = RecurrentNet(input_size, hidden_sizes, recurrent_size, recurrent_depth)
        elif architecture == "LSTM":
            model = LSTMNet(input_size, hidden_sizes, recurrent_size, recurrent_depth, dropout=dropout, func=func)
    if architecture == "MLP":
        model = MLP(input_size, hidden_sizes, func=func, dropout=dropout)
    file_path = join(_package_path, "trained_models/"+model_name+".pth")
    model.load_state_dict(torch.load(file_path))
    model.eval()
    model.to("cpu") 
    model.requires_grad_(False)
    print("model loaded")
    return model, architecture