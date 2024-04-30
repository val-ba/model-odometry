import numpy as np
import torch
from os.path import join
import yaml
import pandas as pd
import pathlib
import argparse

file_path = join(pathlib.Path(__file__).parent.parent.resolve() , "Data/")
data_columns = pd.read_csv(join(str(pathlib.Path(__file__).parent.parent.resolve()),"Model/data_to_use.csv")).columns 
label_columns = pd.read_csv(join(str(pathlib.Path(__file__).parent.parent.resolve()),"Model/labels_to_use.csv")).columns 
def calculate_normalization_parameters(main_df):
    norm_factors = {}
    for column in main_df.columns:
        mean = main_df[column].mean()
        std = main_df[column].std()
        main_df[column] = (main_df[column] - mean)/std
        norm_factors[column] = {"mean": float(mean), "std": float(std)}
    return norm_factors

def normalize_sig(input):
    """normalize using sigmoid function"""
    if isinstance(input, list):
        input = np.array(input)
    return 1/(1+np.exp(-input))

def denormalize_sig(output):
    """denormalize using inverse sigmoid function"""
    if isinstance(output, list):
        output = np.array(output)
    return np.log(output/(1-output))

def normalize(input, col_names=data_columns, dataset="Data"):
    """normalize using z score"""
    try:
        if isinstance(input, list):
            input = np.array(input)
        with open(join(file_path, dataset, "norm_factors.yaml"), "r") as f:
            norm_factors = yaml.load(f, Loader=yaml.FullLoader)
        for i, col in enumerate(col_names):
            mean = norm_factors[col]["mean"]
            std = norm_factors[col]["std"]
            input[:, i] = (input[:, i] - mean)/std
        return input
    except KeyError as e:
        print("KeyError: ", e)
        print("Make sure that the dataset is in the Data folder and that the norm_factors.yaml file is in the dataset folder.")
        print("If you want to create the norm_factors.yaml file, run this script with the dataset name as argument.")
        return None
    
def denormalize(output, col_names=label_columns, dataset="Data"):
    """denormalize using z score"""
    try:
        if isinstance(output, list):
            output = np.array(output)
        if isinstance(output, torch.Tensor):
            output = output.cpu().detach().numpy()
        with open(join(file_path, dataset, "norm_factors.yaml"), "r") as f:
            norm_factors = yaml.load(f, Loader=yaml.FullLoader)
        for i, col in enumerate(col_names):
            mean = norm_factors[col]["mean"]
            std = norm_factors[col]["std"]
            output[:, i] = output[:, i]*std + mean
        return output
    except KeyError as e:
        print("KeyError: ", e)
        print("Make sure that the dataset is in the Data folder and that the norm_factors.yaml file is in the dataset folder.")
        print("If you want to create the norm_factors.yaml file, run this script with the dataset name as argument.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="rosbags_real_world_april", help="Name of dataset")
    parser.add_argument("-c", "--copy", action="store_true", help="Copy the norm factors of dataset with name in --copy_from")
    parser.add_argument("-cf", "--copy_from", type=str, default="rosbags_real_world_april", help="Name of dataset to copy from")
    args = parser.parse_args()
    main_df = pd.read_feather(join(file_path, args.dataset,"data_proc.feather"))
    if not args.copy:
        norm_factors = calculate_normalization_parameters(main_df)
        with open(join(file_path, args.dataset,"norm_factors.yaml"), "w") as f:
            yaml.dump(norm_factors, f)
        print("Normalization factors saved to norm_factors.yaml")
        print("testing, whether normlizing worked:")
        col_names = ["orientation.w", "r_foot_pose.position.z", "phase"]
        input = main_df[col_names].values
        output = normalize(input, col_names, args.dataset)
        assert np.allclose(denormalize(output, col_names, args.dataset), input)
        print("All tests passed.")
    else:
        with open(join(file_path, args.copy_from,"norm_factors.yaml"), "r") as f:
            norm_factors = yaml.load(f, Loader=yaml.FullLoader)
        with open(join(file_path, args.dataset,"norm_factors.yaml"), "w") as f:
            yaml.dump(norm_factors, f)
        print("Normalization factors copied from ", args.copy_from, " to ", args.dataset)
        print("testing, whether normlizing worked:")
        col_names = ["orientation.w", "r_foot_pose.position.z", "phase"]
        input = main_df[col_names].values
        output = normalize(input, col_names, args.dataset)
        assert np.allclose(denormalize(output, col_names, args.dataset), input)
        print("All tests passed.")


    