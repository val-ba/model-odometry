#!/usr/bin/env python
import pandas as pd
from os.path import join
from transforms3d.quaternions import quat2mat
from transforms3d.affines import compose
import numpy as np
import pathlib
import argparse
from matplotlib import pyplot as plt
from transforms_for_odom import get_transforms, get_r2l_and_l2r_transforms

def plot_phase_against_z(ground_truth_df, df):
    l_z = ground_truth_df["l_foot_pose.position.z"]
    r_z = ground_truth_df["r_foot_pose.position.z"]
    phase = df["phase"]
    plt.plot(l_z, label="l_foot_pose.position.z")
    plt.plot(r_z, label="r_foot_pose.position.z")
    plt.plot(phase, label="phase")
    # blue points where phase is -1, otherwise red
    plt.scatter(phase[phase == -1].index, np.linspace(0, 0, len(phase[phase == -1])), c="b")
    plt.scatter(phase[phase == 1].index, np.linspace(0, 0, len(phase[phase == 1])), c="r")
    
    # call x axis meters
    plt.xlabel("foot height in meters")
    plt.ylabel("time in seconds times 1000")
    
    plt.legend()
    # save as pdf
    plt.savefig("phase_against_z.pdf")
    plt.clf()
    
def plot_linear_acc_z_against_z(ground_truth_df, imu_df):
    # we want to first normalize both betwen 0 and 1 to compare them
    ground_truth_df["l_foot_pose.position.z"] = (ground_truth_df["l_foot_pose.position.z"] - ground_truth_df["l_foot_pose.position.z"].min()) / (ground_truth_df["l_foot_pose.position.z"].max() - ground_truth_df["l_foot_pose.position.z"].min())
    ground_truth_df["r_foot_pose.position.z"] = (ground_truth_df["r_foot_pose.position.z"] - ground_truth_df["r_foot_pose.position.z"].min()) / (ground_truth_df["r_foot_pose.position.z"].max() - ground_truth_df["r_foot_pose.position.z"].min())
    imu_df["linear_acceleration.z"] = (imu_df["linear_acceleration.z"] - imu_df["linear_acceleration.z"].min()) / (imu_df["linear_acceleration.z"].max() - imu_df["linear_acceleration.z"].min())
    plt.plot(ground_truth_df["l_foot_pose.position.z"], label="l_foot_pose.position.z")
    plt.plot(ground_truth_df["r_foot_pose.position.z"], label="r_foot_pose.position.z")
    plt.plot(imu_df["linear_acceleration.z"], label="linear_acceleration.z")
    plt.xlabel("time in seconds times 1000")
    plt.ylabel("normalized values")
    plt.legend()
    plt.savefig("linear_acc_z_against_z.pdf")
    plt.clf()
    
def plot_tf_x_y_against_groundtruth(df):
    x_gt = df["/ground_truth.transform_translation.x"]
    x_tf = df["/tf_r_sole_to_l_sole_transform.translation.x"]
    y_gt =df["/ground_truth.transform_translation.y"]
    y_tf = df["/tf_r_sole_to_l_sole_transform.translation.y"]
    x2_tf = df["/tf_l_sole_to_r_sole_transform.translation.x"]
    y2_tf = df["/tf_l_sole_to_r_sole_transform.translation.y"]
    fig, axs = plt.subplots(2)
    axs[0].plot(x_gt, label="x ground truth", c="r")
    axs[0].plot(x_tf, label="x tf", c="b")
    axs[0].plot(x2_tf, label="x2 tf", c="g")
    axs[1].plot(y_gt, label="y ground truth", c="r")
    axs[1].plot(y_tf, label="y tf", c="b")
    axs[1].plot(y2_tf, label="y2 tf", c="g")
    plt.legend()
    plt.savefig("tf_x_y_against_groundtruth.pdf")
    plt.clf()

def plot_ground_truth_feet(main_df):
    fig, ax = plt.subplots()
    ax.plot(main_df["r_foot_pose.position.x"].to_numpy(), main_df["r_foot_pose.position.y"].to_numpy(), label='r_foot_pose')
    ax.plot(main_df["l_foot_pose.position.x"].to_numpy(), main_df["l_foot_pose.position.y"].to_numpy(), label='l_foot_pose')
    ax.set(xlabel='x', ylabel='y', title='Robot position')
    ax.legend()
    ax.grid()
    plt.savefig("ground_truth_feet.pdf")
    print(f"Saved ground truth feet plot to ground_truth_feet.pdf")
    plt.clf()

def get_r2l_and_l2r_transforms(r_foot_translation, r_foot_orientation, l_foot_translation, l_foot_orientation):
    '''Computes the translation and rotation from right foot to left foot and vice versa.
    The input should be a numpy array of shape (n, 3) for translation and (n, 4) for orientation. (xyzw)'''
    r_foot_orientation = [quat2mat([o[3], o[0], o[1], o[2]]) for o in r_foot_orientation]
    l_foot_orientation = [quat2mat([o[3], o[0], o[1], o[2]]) for o in l_foot_orientation]
    world_to_r_foot = [compose(trans, rot, np.ones(3)) for trans, rot in zip(r_foot_translation, r_foot_orientation)]
    world_to_l_foot = [compose(trans, rot, np.ones(3)) for trans, rot in zip(l_foot_translation, l_foot_orientation)]
    r_to_l_translation, r_to_l_yaw = get_transforms(world_to_r_foot, world_to_l_foot)
    l_to_r_translation, l_to_r_yaw = get_transforms(world_to_l_foot, world_to_r_foot)
    return r_to_l_translation, r_to_l_yaw, l_to_r_translation, l_to_r_yaw

def plot_foot_x_against_angular_velocity_x(ground_truth_df, imu_df, main_df):
    r_foot_translation = ground_truth_df[["r_foot_pose.position.x", "r_foot_pose.position.y", "r_foot_pose.position.z"]].to_numpy()
    l_foot_translation = ground_truth_df[["l_foot_pose.position.x", "l_foot_pose.position.y", "l_foot_pose.position.z"]].to_numpy()
    r_foot_orientation = ground_truth_df[["r_foot_pose.orientation.x", "r_foot_pose.orientation.y", "r_foot_pose.orientation.z", "r_foot_pose.orientation.w"]].to_numpy()
    l_foot_orientation = ground_truth_df[["l_foot_pose.orientation.x", "l_foot_pose.orientation.y", "l_foot_pose.orientation.z", "l_foot_pose.orientation.w"]].to_numpy()
    r_to_l_t,  _, __, ___ = get_r2l_and_l2r_transforms(r_foot_translation, r_foot_orientation, l_foot_translation, l_foot_orientation)
    
    r_to_l_x_norm = (r_to_l_t[:,0] - r_to_l_t[:,0].min()) / (r_to_l_t[:,0].max() - r_to_l_t[:,0].min())
    plt.plot(r_to_l_x_norm, label="r_to_l_x_norm")

    imu_df["angular_velocity.x"] = (imu_df["angular_velocity.x"] - imu_df["angular_velocity.x"].min()) / (imu_df["angular_velocity.x"].max() - imu_df["angular_velocity.x"].min())
    plt.plot(imu_df["angular_velocity.x"], label="angular_velocity.x")
    
    plt.xlabel("time in seconds times 1000")
    plt.ylabel("normalized values")
    plt.legend()
    plt.savefig("foot_x_against_angular_velocity_x.pdf")
    plt.clf()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default="rosbags", help="Name of rosbag folder")
    parser.add_argument("-n", "--number", type=int, default=1000, help="Number of the file to process")
    parser.add_argument("-f", "--feet", action="store_true", help="Plot the ground truth feet")
    parser.add_argument("-zp", "--z_phase", action="store_true", help="Plot the phase against z")
    parser.add_argument("-za", "--z_acc", action="store_true", help="Plot the linear acceleration z against z") 
    parser.add_argument("-tf", "--tf_xy", action="store_true", help="Plot the tf x y against ground truth")
    parser.add_argument("-fax", "--foot_x_angular_velocity_x", action="store_true", help="Plot the foot x against angular velocity x")
    args = parser.parse_args()
    data_folder = args.path
    file_path = join(pathlib.Path(__file__).parent.parent.resolve() , "Data" , data_folder, "data_proc.feather")
    ground_truth_path = join(pathlib.Path(__file__).parent.parent.resolve() , "Data" , data_folder, "__model_states_rosbag_data.feather")
    imu_path = join(pathlib.Path(__file__).parent.parent.resolve() , "Data" , data_folder, "__imu__data_rosbag_data.feather")
    tf_l2r_path = join(pathlib.Path(__file__).parent.parent.resolve() , "Data" , data_folder, "tf_l2r_rosbag_data.feather")
    ground_truth_df = pd.read_feather(ground_truth_path)
    imu_df = pd.read_feather(imu_path)
    df = pd.read_feather(file_path)
    df = df.head(args.number)
    last_time_stamp = df.index[-1]
    ground_truth_df = ground_truth_df[ground_truth_df.index <= last_time_stamp]
    imu_df = imu_df[imu_df.index <= last_time_stamp]
    
    if args.feet:
        plot_ground_truth_feet(ground_truth_df)
    if args.z_phase:
        plot_phase_against_z(ground_truth_df, df)
    if args.z_acc:
        plot_linear_acc_z_against_z(ground_truth_df, imu_df)
    if args.tf_xy:
        plot_tf_x_y_against_groundtruth(df)
    if args.foot_x_angular_velocity_x:
        plot_foot_x_against_angular_velocity_x(ground_truth_df, imu_df, df)
        
    print("Done plotting.")
        
    

    
    
    