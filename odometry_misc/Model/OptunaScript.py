import torch
import optuna
from OdometryDataset import OdomDataset
import tqdm
from os.path import join
import pathlib
from optuna.trial import TrialState
import argparse
from train_model import train_loop, test_loop, initialize_training, make_train_and_test_loader
import traceback
parser = argparse.ArgumentParser(description='Load Optuna study')
parser.add_argument('--study_name', "-n", type=str, default="odometry_study_default", help='Name of the study')
parser.add_argument('--trials', "-t", type=int, default=1000, help='Number of trials')
parser.add_argument("-p", "--path", type=str, default="Data", help="Path to the data for training")
args = parser.parse_args()
data_path = join(str(pathlib.Path(__file__).parent.parent.resolve()),"Data", args.path)
print(f"Data path is {data_path}")


print("Creating a sequential data set. This will take a while...")
torch.manual_seed(42)
train_recurrent_dataset_whole = OdomDataset(data_path, sequence=5)
recurrent_train_size = len(train_recurrent_dataset_whole)
train_recurrent_dataset, validation_recurrent_set = torch.utils.data.random_split(train_recurrent_dataset_whole, [int(recurrent_train_size*0.8),recurrent_train_size-int(recurrent_train_size*0.8)])
print("Creating a non-sequential data set. This will take a bit...")
train_dataset_whole = OdomDataset(data_path, sequence=1)
train_size = len(train_dataset_whole)
train_dataset, validation_set = torch.utils.data.random_split(train_dataset_whole, [int(train_size*0.8), train_size-int(train_size*0.8)])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
        

def define_layers(trial):
    number_of_layers = trial.suggest_int('number_of_layers', 1, 5)
    layers = []
    for i in range(number_of_layers):
        number_of_output_features = trial.suggest_int('n_units_l{}'.format(i), 4, 128)
        layers.append(number_of_output_features)
    layers = "|".join([str(layer) for layer in layers])
    return layers


def objective(trial):
    dropout = trial.suggest_float('dropout', 0.01, 0.15)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    epochs = trial.suggest_int('epochs', 10, 100)
    architecture = trial.suggest_categorical('architecture', ["MLP", "RNN", "LSTM"])
    criterion = trial.suggest_categorical('criterion', ["mse", "mae"])
    optimizer = trial.suggest_categorical('optimizer', ["adam", "sgd"])
    recurrent_size = trial.suggest_int('recurrent_size', 4, 64)
    recurrent_depth = trial.suggest_int('recurrent_depth', 1, 16)
    activation_function = trial.suggest_categorical('activation_function', ["relu", "tanh", "sigmoid"])
    hidden_sizes = define_layers(trial)
    batch_size = trial.suggest_int('batch_size', 128, 512)
    
    if architecture == "RNN" or architecture == "LSTM":
        trainloader, testloader = make_train_and_test_loader(train_recurrent_dataset, batch_size, different_val_set=validation_recurrent_set)
        dset = train_recurrent_dataset_whole
    else:
        trainloader, testloader = make_train_and_test_loader(train_dataset, batch_size, different_val_set=validation_set)
        dset = train_dataset_whole
    

    net, criterion, optimizer = initialize_training(architecture, dset, hidden_sizes, dropout, criterion, optimizer, learning_rate, activation_function, recurrent_size, recurrent_depth)
    for t in tqdm.tqdm(range(epochs)):
        try:
            net.to(device)
            train_loop(trainloader, net, criterion, optimizer, architecture)
            # shuffle train data after each epoch
            net.eval()
            test_loss = test_loop(testloader, net, criterion, architecture, return_loss=True)
            trial.report(test_loss, t)
            if trial.should_prune():
                raise optuna.TrialPruned()
            net.train()
        except Exception as e:
            print(e)
            traceback.print_exc()
            raise optuna.TrialPruned()
    

    return test_loss



if __name__ == "__main__":



    study_name = parser.parse_args().study_name

    study = optuna.create_study(direction="minimize", study_name=study_name, storage="sqlite:///"+study_name+".db", load_if_exists=True)
    study.optimize(objective, n_trials=parser.parse_args().trials)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    print(f"{recurrent_train_size=}")
    print(f"{train_size=}")

    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_slice(study).show()
    optuna.visualization.plot_parallel_coordinate(study).show()
    optuna.visualization.plot_contour(study, params=["dropout", "learning_rate"]).show()
    optuna.visualization.plot_param_importances(study).show()
    optuna.visualization.plot_edf(study).show()
    optuna.visualization.plot_intermediate_values(study).show()
    optuna.visualization.plot_optimization_history(study).show()
