#!/usr/bin/env python
import pandas as pd
from os import listdir
from os.path import join
from transforms3d.quaternions import mat2quat
import time
import datetime
import pathlib
import argparse
import os
import numpy as np
import transforms3d as tf3d

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str, default="rosbags_real_world_april", help="Name of tsv folder")
parser.add_argument("-c", "--cores", type=int, default=8, help="Number of cores to use")
parser.add_argument("-csv", "--csv", action="store_true", help="Save as csv")
args = parser.parse_args()
data_path = join(pathlib.Path(__file__).parent.parent.resolve(), "Data/",args.path+"/")

dir_path = path = join(os.path.expanduser("~"),args.path)
time_stamp = None
main_df = None

tsv_files = [f for f in listdir(dir_path) if f.endswith('_6D.tsv')]

for file in tsv_files:
    path = join(dir_path, file)
    with open(path) as f:
        content = f.readlines()
        # the time stamp is in line 8
        time_stamp = content[7].split('\t')[1]
        time_stamp = time.mktime(datetime.datetime.strptime(time_stamp, "%Y-%m-%d, %H:%M:%S.%f").timetuple())
        # the first 13 columns are just info, but not data
        content = content[13:]
        with open(file, 'w') as f:
            for line in content:
                f.write(line)
    
    run_number = int(file.split('_')[0][3:])
    df=pd.read_table(file, sep='\t')
    df['Time'] = time_stamp + df['Time'].astype(float)    
    df['header.stamp.sec'] = df['Time'].astype(int)
    df['header.stamp.nanosec'] = df['Time'].apply(lambda x: ((x - int(x) )* 1e9))
    df['header.stamp.nanosec'] = df['header.stamp.nanosec'].astype(int)
    df['bag_number'] = run_number
    if main_df is None:
        main_df = df
    else:
        main_df = pd.concat([main_df, df])
    
# change column name from r_foot X to r_foot_pose.position.x
main_df = main_df.rename(columns={'r_foot X': 'r_foot_pose.position.x', 'Y.1': 'r_foot_pose.position.y', 'Z.1': 'r_foot_pose.position.z', 
                                  'l_foot X': 'l_foot_pose.position.x', 'Y': 'l_foot_pose.position.y', 'Z': 'l_foot_pose.position.z',
                                  'head X': 'base_link_pose.position.x', 'Y.2': 'base_link_pose.position.y', 'Z.2': 'base_link_pose.position.z'})
# we want to calculate less data, because it is high in frequency
main_df = main_df.iloc[::10, :]
main_df = main_df[main_df["r_foot_pose.position.x"] != 0]
main_df = main_df[main_df["l_foot_pose.position.x"] != 0]

l_rot_mat = main_df[["Rot[0]", "Rot[1]", "Rot[2]", "Rot[3]", "Rot[4]", "Rot[5]", "Rot[6]", "Rot[7]", "Rot[8]"]].to_numpy()
r_rot_mat = main_df[["Rot[0].1", "Rot[1].1", "Rot[2].1", "Rot[3].1", "Rot[4].1", "Rot[5].1", "Rot[6].1", "Rot[7].1", "Rot[8].1"]].to_numpy()
h_rot_mat = main_df[["Rot[0].2", "Rot[1].2", "Rot[2].2", "Rot[3].2", "Rot[4].2", "Rot[5].2", "Rot[6].2", "Rot[7].2", "Rot[8].2"]].to_numpy()

l_foot_translation = main_df[["l_foot_pose.position.x", "l_foot_pose.position.y", "l_foot_pose.position.z"]].to_numpy()
r_foot_translation = main_df[["r_foot_pose.position.x", "r_foot_pose.position.y", "r_foot_pose.position.z"]].to_numpy()
head_translation = main_df[["base_link_pose.position.x", "base_link_pose.position.y", "base_link_pose.position.z"]].to_numpy()

l_foot_orientation = [mat2quat(np.array([[r[0], r[3], r[6]], [r[1], r[4], r[7]], [r[2], r[5], r[8]]])) for r in l_rot_mat]
r_foot_orientation = [mat2quat(np.array([[r[0], r[3], r[6]], [r[1], r[4], r[7]], [r[2], r[5], r[8]]])) for r in r_rot_mat]
head_orientation = [mat2quat(np.array([[r[0], r[3], r[6]], [r[1], r[4], r[7]], [r[2], r[5], r[8]]])) for r in h_rot_mat]

# create transform which goes -0.3 on the z-axis

z_down_transform = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.3], [0, 0, 0, 1]])
# apply the transform to the translation of head, because we want to measure the base link
head_translation = np.array([z_down_transform @ np.array([t[0], t[1], t[2], 1]) for t in head_translation])
head_translation = head_translation[:, :3]
# set head translation to the main_df
main_df[["base_link_pose.position.x", "base_link_pose.position.y", "base_link_pose.position.z"]] = head_translation

main_df.insert(1, 'r_foot_pose.orientation.x', [r[1] for r in r_foot_orientation])
main_df.insert(2, 'r_foot_pose.orientation.y', [r[2] for r in r_foot_orientation])
main_df.insert(3, 'r_foot_pose.orientation.z', [r[3] for r in r_foot_orientation])
main_df.insert(4, 'r_foot_pose.orientation.w', [r[0] for r in r_foot_orientation])

main_df.insert(5, 'l_foot_pose.orientation.x', [r[1] for r in l_foot_orientation])
main_df.insert(6, 'l_foot_pose.orientation.y', [r[2] for r in l_foot_orientation])
main_df.insert(7, 'l_foot_pose.orientation.z', [r[3] for r in l_foot_orientation])
main_df.insert(8, 'l_foot_pose.orientation.w', [r[0] for r in l_foot_orientation])

main_df.insert(9, 'base_link_pose.orientation.x', [r[1] for r in head_orientation])
main_df.insert(10, 'base_link_pose.orientation.y', [r[2] for r in head_orientation])
main_df.insert(11, 'base_link_pose.orientation.z', [r[3] for r in head_orientation])
main_df.insert(12, 'base_link_pose.orientation.w', [r[0] for r in head_orientation])
ground_truth_turned = tf3d.quaternions.qmult(tf3d.quaternions.axangle2quat([0, 0, 1], np.pi/2), main_df[["base_link_pose.orientation.w", "base_link_pose.orientation.x", "base_link_pose.orientation.y", "base_link_pose.orientation.z"]].to_numpy().T).T
main_df["base_link_pose.orientation.w"] = ground_truth_turned[:,0]
main_df["base_link_pose.orientation.x"] = ground_truth_turned[:,1]
main_df["base_link_pose.orientation.y"] = ground_truth_turned[:,2]
main_df["base_link_pose.orientation.z"] = ground_truth_turned[:,3]

main_df['time'] = main_df['header.stamp.sec'] + main_df['header.stamp.nanosec'] * 1e-9
cols_to_keep = ['r_foot_pose.position.x', 'r_foot_pose.position.y', 'r_foot_pose.position.z', 
                'r_foot_pose.orientation.x', 'r_foot_pose.orientation.y', 'r_foot_pose.orientation.z', 'r_foot_pose.orientation.w', 'l_foot_pose.position.x', 
                'l_foot_pose.position.y', 'l_foot_pose.position.z', 'l_foot_pose.orientation.x', 'l_foot_pose.orientation.y', 'l_foot_pose.orientation.z', 
                'l_foot_pose.orientation.w', "bag_number", "base_link_pose.position.x", "base_link_pose.position.y", "base_link_pose.position.z",
                "base_link_pose.orientation.x", "base_link_pose.orientation.y", "base_link_pose.orientation.z", "base_link_pose.orientation.w", "time"]
cols_to_drop = main_df.columns.to_list()
for col in cols_to_keep:
    cols_to_drop.remove(col)
main_df.drop(cols_to_drop, axis=1, inplace=True)
# sort by first, bag_number, then time, this functions as unique key of the table
main_df = main_df.sort_values(by=["bag_number", "time"])
main_df = main_df.reset_index(drop=True)

# meters
main_df[["r_foot_pose.position.x", "r_foot_pose.position.y", "r_foot_pose.position.z", 
         "l_foot_pose.position.x", "l_foot_pose.position.y", "l_foot_pose.position.z", "base_link_pose.position.x", "base_link_pose.position.y", "base_link_pose.position.z"]] /= 1000

# remove all rows, where r_foot_pose.position.z is above 0.08, because then the robot is not moving, but was lifted up
main_df = main_df[main_df["r_foot_pose.position.z"] < 0.08]

# we give it the same name as the model states from simulation
file_name = "__model_states_rosbag_data" 
if not os.path.exists(data_path):
    os.makedirs(data_path)
if args.csv:
    main_df.to_csv(data_path+file_name+ ".csv")
main_df.to_feather(data_path+file_name+".feather")

print("data path is: ", data_path)

print("done")




