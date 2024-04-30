#!/usr/bin/env python
import pandas as pd
from os import listdir
from os.path import join
from transforms3d.euler import quat2euler
import pathlib
import argparse
from transforms_for_odom import get_r2l_and_l2r_transforms


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str, default="rosbags_sole_random_3", help="Name of rosbag folder")
parser.add_argument("-csv", "--csv", action="store_true", help="Save as csv")
parser.add_argument("-r", "--real_world", action="store_true", help="Use real world data")
args = parser.parse_args()

data_path = join(pathlib.Path(__file__).parent.parent.resolve(), "Data/",args.path+"/")
files = [f for f in listdir(data_path) if f.endswith('.feather')]
print(f"Found the following files: {files}")
processed_files = [f for f in files if "proc" in f]
print(f"Found the following processed files: {processed_files}")

def get_steps_from_sim(dataframe):
    """Computes the ground contact from simulation data.
    -1 means a switch from right to left and 1 means a switch from left to right"""
    assert "l_foot_pose.position.z" in dataframe.columns, "l_foot_pose.position.z not in dataframe"
    assert "r_foot_pose.position.z" in dataframe.columns, "r_foot_pose.position.z not in dataframe"
    
    dataframe["step"] = -1
    dataframe.loc[dataframe[["l_foot_pose.position.z", "r_foot_pose.position.z"]].min(axis=1) == dataframe["l_foot_pose.position.z"], "step"] = 0
    dataframe.loc[dataframe[["l_foot_pose.position.z", "r_foot_pose.position.z"]].min(axis=1) == dataframe["r_foot_pose.position.z"], "step"] = 1
    # add phase switch when step changes

    dataframe["phase"] = dataframe["step"].diff()
    phase_times = dataframe.loc[dataframe["phase"] != 0].dropna()
    print(f"Found the following indices for phase switches: {phase_times.index}")
    print("They should be jumping around a bit")
    return phase_times

def save_to_feather_and_csv(dataframe, name):
    dataframe.to_feather(data_path+name+"_proc.feather")
    if args.csv:
        dataframe.to_csv(data_path+name+"_proc.csv")

start_time = pd.Timestamp.now()
ground_truth_file = [f for f in files if "model_state" in f and f.endswith(".feather")]
l2r_tf = [f for f in files if "tf" in f and "l2r" in f and f.endswith(".feather")]
r2l_tf = [f for f in files if "tf" in f and "r2l" in f and f.endswith(".feather")]
imu_file = [f for f in files if "imu" in f and f.endswith(".feather")]
r2b_file = [f for f in files if "right_to_base" in f and f.endswith(".feather")]
l2b_file = [f for f in files if "left_to_base" in f and f.endswith(".feather")]
motion_odometry_file = [f for f in files if "motion_odometry" in f and f.endswith(".feather")]
walk_engine_file = [f for f in files if "walk_engine" in f and f.endswith(".feather")]
print("Loading files.")

ground_truth_frame = pd.read_feather(data_path+ground_truth_file[0])
imu_frame = pd.read_feather(data_path+imu_file[0])
l2r_tf_frame = pd.read_feather(data_path+l2r_tf[0])
r2l_tf_frame = pd.read_feather(data_path+r2l_tf[0])
r2b_frame = pd.read_feather(data_path+r2b_file[0])
l2b_frame = pd.read_feather(data_path+l2b_file[0])
motion_odometry = pd.read_feather(data_path+motion_odometry_file[0])
walk_engine = pd.read_feather(data_path+walk_engine_file[0])

ground_truth_frame = ground_truth_frame.sort_values(by=["bag_number", "time"]).head(ground_truth_frame.shape[0]//k)
imu_frame = imu_frame.sort_values(by=["bag_number", "time"]).head(imu_frame.shape[0]//k)
l2r_tf_frame = l2r_tf_frame.sort_values(by=["bag_number", "time"]).head(l2r_tf_frame.shape[0]//k)
r2l_tf_frame = r2l_tf_frame.sort_values(by=["bag_number", "time"]).head(r2l_tf_frame.shape[0]//k)
r2b_frame = r2b_frame.sort_values(by=["bag_number", "time"]).head(r2b_frame.shape[0]//k)
l2b_frame = l2b_frame.sort_values(by=["bag_number", "time"]).head(l2b_frame.shape[0]//k)
motion_odometry = motion_odometry.sort_values(by=["bag_number", "time"]).head(motion_odometry.shape[0]//k)
walk_engine = walk_engine.sort_values(by=["bag_number", "time"]).head(walk_engine.shape[0]//k)

walk_engine.columns = [f"walk_{column}" if column not in ["bag_number", "time"] else column for column in walk_engine.columns]
motion_odometry.columns = [f"motion_{column}" if column not in ["bag_number", "time"] else column for column in motion_odometry.columns]

print("concatenating files.")


# outer join on time and bag_number
combined_df = pd.merge(ground_truth_frame, imu_frame, on=['bag_number', "time"], how='outer')
combined_df = pd.merge(combined_df, l2r_tf_frame, on=['bag_number', "time"], how='outer')
combined_df = pd.merge(combined_df, r2l_tf_frame, on=['bag_number', "time"], how='outer')
combined_df = pd.merge(combined_df, r2b_frame, on=['bag_number', "time"], how='outer')
combined_df = pd.merge(combined_df, l2b_frame, on=['bag_number', "time"], how='outer')
combined_df = pd.merge(combined_df, motion_odometry, on=['bag_number', "time"], how='outer')
combined_df = pd.merge(combined_df, walk_engine, on=['bag_number', "time"], how='outer')
combined_df = combined_df.sort_values(by=["bag_number", "time"]).ffill()
combined_df.dropna(inplace=True)
combined_df.drop_duplicates(subset=["time", "bag_number"], inplace=True)
combined_frame = combined_df.reset_index(drop=True)



print("Computing yaw.")
left_to_right_yaw = combined_frame[["/tf_l_sole_to_r_sole_transform.rotation.w", "/tf_l_sole_to_r_sole_transform.rotation.x", "/tf_l_sole_to_r_sole_transform.rotation.y", "/tf_l_sole_to_r_sole_transform.rotation.z"]].to_numpy()
combined_frame["/tf_l2r_yaw"] = [quat2euler(o)[2] for o in left_to_right_yaw]
right_to_left_yaw = combined_frame[["/tf_r_sole_to_l_sole_transform.rotation.w", "/tf_r_sole_to_l_sole_transform.rotation.x", "/tf_r_sole_to_l_sole_transform.rotation.y", "/tf_r_sole_to_l_sole_transform.rotation.z"]].to_numpy()
combined_frame["/tf_r2l_yaw"] = [quat2euler(o)[2] for o in right_to_left_yaw]

print("Getting steps.")
phases_frame = get_steps_from_sim(combined_frame)
combined_frame["phase"] = phases_frame["phase"]

combined_with_odoms = combined_frame.drop_duplicates(subset=["time", "bag_number"])
combined_with_odoms = combined_with_odoms.reset_index(drop=True)

save_to_feather_and_csv(combined_with_odoms, "combined_with_odoms")


combined_frame = combined_frame.loc[phases_frame.index]
r_foot_translation = combined_frame[["r_foot_pose.position.x", "r_foot_pose.position.y", "r_foot_pose.position.z"]].to_numpy()
l_foot_translation = combined_frame[["l_foot_pose.position.x", "l_foot_pose.position.y", "l_foot_pose.position.z"]].to_numpy()
r_foot_orientation = combined_frame[["r_foot_pose.orientation.x", "r_foot_pose.orientation.y", "r_foot_pose.orientation.z", "r_foot_pose.orientation.w"]].to_numpy()
l_foot_orientation = combined_frame[["l_foot_pose.orientation.x", "l_foot_pose.orientation.y", "l_foot_pose.orientation.z", "l_foot_pose.orientation.w"]].to_numpy()
r_to_l_translation, r_to_l_yaw, l_to_r_translation, l_to_r_yaw = get_r2l_and_l2r_transforms(r_foot_translation, r_foot_orientation, l_foot_translation, l_foot_orientation)

combined_frame["/ground_truth.r_to_l_translation.x"] = r_to_l_translation[:,0]
combined_frame["/ground_truth.r_to_l_translation.y"] = r_to_l_translation[:,1]
combined_frame["/ground_truth.r_to_l_yaw"] = r_to_l_yaw

combined_frame["/ground_truth.l_to_r_translation.x"] = l_to_r_translation[:,0]
combined_frame["/ground_truth.l_to_r_translation.y"] = l_to_r_translation[:,1]
combined_frame["/ground_truth.l_to_r_yaw"] = l_to_r_yaw

right_to_left_switches = combined_frame.loc[phases_frame[phases_frame["phase"] == -1].index]
left_to_right_switches = combined_frame.loc[phases_frame[phases_frame["phase"] == 1].index]

right_to_left_switches = right_to_left_switches[["/ground_truth.r_to_l_translation.x", "/ground_truth.r_to_l_translation.y", "/ground_truth.r_to_l_yaw","bag_number", "time"]]
left_to_right_switches = left_to_right_switches[["/ground_truth.l_to_r_translation.x", "/ground_truth.l_to_r_translation.y", "/ground_truth.l_to_r_yaw","bag_number", "time"]]
right_to_left_switches.columns = [column.replace("r_to_l", "transform") for column in right_to_left_switches.columns]
left_to_right_switches.columns = [column.replace("l_to_r", "transform") for column in left_to_right_switches.columns]

label_frame = pd.concat([right_to_left_switches, left_to_right_switches], axis=0)
label_frame.sort_values(by=["bag_number", "time"], inplace=True)
label_frame = label_frame[["/ground_truth.transform_translation.x", "/ground_truth.transform_translation.y", "/ground_truth.transform_yaw"]]


combined_frame = pd.merge(combined_frame, label_frame, left_index=True, right_index=True)


save_to_feather_and_csv(combined_frame, "data")

print(f"Finished processing in {pd.Timestamp.now() - start_time}")









    




