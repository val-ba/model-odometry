from torch.utils.data import Dataset
import pandas as pd
from os.path import join
import os
import pathlib
import numpy as np
from multiprocessing import Pool
import pickle
import torch
from scripts.normalize_sample import normalize
import matplotlib.pyplot as plt
from multiprocessing import Pool

class OdomDataset(Dataset):
    """
    Dataset for odometry data
    Use static method get_topic_dict to get a dict with all topics and their sample size and stride
    and modify it to your liking.
    args: path: path to the folder with the feather files

    """
    def __init__(self, folder, sequence=1, size=-1):
        """_summary_

        Args:
            folder: path to the folder with the feather files, does not needs to be absolute.
            sequence (int, optional): If sequence is 1, it is not split. If sequence is -1 it is the maximal split. Defaults to 1.
            time_stamps (bool, optional): _description_. Defaults to False.
        """
        self.labels, self.data, self.label_names, self.data_names = self.make_labels_and_data(folder, sequence)
        # add end_of_bag to data_names

        if size>0:
            self.labels = self.labels[:size]
            self.data = self.data[:size]
        self.len = self.labels.shape[0]


    def make_labels_and_data(self, path, sequence):
        absolute_path = join(str(pathlib.Path(__file__).parent.parent.resolve()),"Data/",path)
        df = pd.read_feather(join(absolute_path,"data_proc.feather")).reset_index(drop=True)
        data_columns = pd.read_csv(join(str(pathlib.Path(__file__).parent.resolve()),"data_to_use.csv")).columns 
        label_columns = pd.read_csv(join(str(pathlib.Path(__file__).parent.resolve()),"labels_to_use.csv")).columns 
        data_df = df[data_columns]
        label_df = df[label_columns]
        start_of_bag = df[["bag_number"]].diff() != 0
        label_names = label_df.columns
        data_names = data_df.columns

        labels = label_df.to_numpy().astype(np.float32)
        data = data_df.to_numpy().astype(np.float32)
        labels = normalize(labels, label_names, path)
        data = normalize(data, data_names, path)
        # get columns "end_of_bag" 
        assert len(data_df.index) == len(label_df.index)
        if sequence != 1:
            labels, data = self.create_context(data, labels, sequence, start_of_bag)
        return labels, data, label_names, data_names

    def split_array_sliding_window(self, array, sequence, bag_change_index):
        return [array[i:i+sequence] for i in range(0, len(array)- sequence) if not any(j in bag_change_index for j in  range(i, i+sequence))]
    
    def create_context(self, data, labels, sequence, start_of_bag):
        # bag change time is when the column "end_of_bag" is true
        bag_change_index = start_of_bag.index[start_of_bag["bag_number"] == True].tolist()
        cores = 1 # change if you want to use more cores
        print(f"I need to create a sliding window. This will take a while... I am using {cores} cores.")
        slit_array = np.array_split(data, cores)
        pool = Pool(cores)
        data_bags = pool.starmap(self.split_array_sliding_window, [(array, sequence, bag_change_index) for array in slit_array])
        pool.close()
        pool.join()
        data_bags = np.concatenate(data_bags)
        slit_array = np.array_split(labels, cores)
        pool = Pool(cores)
        label_bags = pool.starmap(self.split_array_sliding_window, [(array, sequence, bag_change_index) for array in slit_array])
        pool.close()
        pool.join()
        label_bags = np.concatenate(label_bags)
        # cast to numpy
        data = np.array(data_bags).astype(np.float32)
        labels = np.array(label_bags).astype(np.float32)
        return labels, data

    def get_labels_and_data(self):
        return self.label_names, self.data_names

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        label = torch.tensor(label).float()
        data = torch.tensor(data).float()
        return label, data

#### TEST ####
if __name__=="__main__":
    path = join(str(pathlib.Path(__file__).parent.parent.resolve()),"Data/rosbags")
    dataset = OdomDataset(path, sequence=1)
    print("Size of dataset input features:")
    print(len(dataset[0]))
    print(dataset.get_labels_and_data())
    print("Now a dataset with sequence")
    dataset = OdomDataset(path, sequence=10)
    print(len(dataset[0]))
    print(dataset.get_labels_and_data())
    print("Now a dataset with maximal sequence")
    dataset = OdomDataset(path, sequence=-1)
    print(dataset[0])
    print(dataset.get_labels_and_data())
    print(dataset[1])

    



