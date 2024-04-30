#!/usr/bin/env python
from rosbags.typesys import get_types_from_msg, register_types
import pandas as pd
from os import listdir
from os.path import join
from rosbags.dataframe import get_dataframe
from rosbags.highlevel import AnyReader
from pathlib import Path
from transforms3d.quaternions import quat2mat
from transforms3d.euler import mat2euler
import os
import numpy as np
import argparse
from multiprocessing import Pool, Value, Lock, Manager
import time
import tf2_ros
from rclpy.time import Duration, Time
from geometry_msgs.msg import TransformStamped
import pathlib
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str, default="rosbags", help="Name of rosbag folder")
parser.add_argument("-c", "--cores", type=int, default=8, help="Number of cores to use")
parser.add_argument("-r", "--real_world", action="store_true", help="Real world data")
parser.add_argument("-m", "--model_states", action="store_true", help="Model states")
parser.add_argument("-f", "--foot_pressure", action="store_true", help="Foot pressure")
parser.add_argument("-w", "--walk_support_phase", action="store_true", help="Walk support phase")
parser.add_argument("-i", "--imu", action="store_true", help="IMU data")
parser.add_argument("-wo", "--walk_odometry", action="store_true", help="Walk odometry")
parser.add_argument("-mo", "--motion_odometry", action="store_true", help="Motion odometry")
parser.add_argument("-tf", "--tf", action="store_true", help="tf data")
parser.add_argument("-csv", "--csv", action="store_true", help="Save as csv")
parser.add_argument("-mod", "--model_odometry", action="store_true", help="Model odometry")


args = parser.parse_args()
Processing_start_time = time.time()
print(args)
def quat2theta(w, x, y, z):
    return mat2euler(quat2mat([w, x, y, z]))[2]

class Counter(object):
    def __init__(self):
        self.val = Value('i', 0)

    def increment(self, n=1):
        with self.val.get_lock():
            self.val.value += n

    @property
    def value(self):
        return self.val.value
    

STRIDX_MSG = """
std_msgs/Header header

geometry_msgs/Pose base_link_pose
geometry_msgs/Twist base_link_twist

geometry_msgs/Pose l_foot_pose

geometry_msgs/Pose r_foot_pose

"""

PHS_MSG = """
std_msgs/Header header
int32 phase
"""
register_types(get_types_from_msg(STRIDX_MSG, 'bitbots_msgs/msg/SimInfo'))
register_types(get_types_from_msg(PHS_MSG, 'biped_interfaces/msg/Phase'))


data_path = join(pathlib.Path(__file__).parent.parent.resolve(), "Data/",args.path+"/")


if not os.path.exists(data_path):
    os.makedirs(data_path)
path = join(os.path.expanduser("~"),args.path)
rosbags = listdir(path)
rosbags = [bag for bag in rosbags if not bag.endswith(".tsv")]
general_fields = ["header.stamp.sec","header.stamp.nanosec"]
nav_msg_fields = ["pose.pose.position.x", "pose.pose.position.y", "pose.pose.position.z", "pose.pose.orientation.x", "pose.pose.orientation.y", "pose.pose.orientation.z", "pose.pose.orientation.w"]
imu_msg_fields = ["angular_velocity.x", "angular_velocity.y", "angular_velocity.z", "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z", "orientation.x", "orientation.y", "orientation.z", "orientation.w"]
generate_model_states_msg_fields = lambda x: [x + ".position.x", x + ".position.y", x + ".position.z", x + ".orientation.x", x + ".orientation.y", x + ".orientation.z", x + ".orientation.w"]
topics = { 
    "/imu/data": general_fields+imu_msg_fields,
    "/motion_odometry": general_fields+nav_msg_fields,
    "/model_states": general_fields+ generate_model_states_msg_fields("base_link_pose")+ generate_model_states_msg_fields("l_foot_pose")+ generate_model_states_msg_fields("r_foot_pose"),
    "/foot_pressure/walk_support_state": general_fields+["phase"],
    "/walk_engine_odometry": general_fields+ nav_msg_fields,
    "/model_odometry": general_fields+ nav_msg_fields,
    "/foot_pressure_left/filtered": general_fields+["left_front", "left_back", "right_front", "right_back"],
    "/foot_pressure_right/filtered": general_fields+["left_front", "left_back", "right_front", "right_back"],
    "/start_end": ["data"]
          }
# add for real world processing the extra topic
if not args.model_states:
    del topics["/model_states"]
if not args.foot_pressure:
    del topics["/foot_pressure_left/filtered"]
    del topics["/foot_pressure_right/filtered"]
if not args.walk_support_phase:
    del topics["/foot_pressure/walk_support_state"]
if not args.imu:
    del topics["/imu/data"]
if not args.walk_odometry:
    del topics["/walk_engine_odometry"]
if not args.motion_odometry:
    del topics["/motion_odometry"]
if not args.model_odometry:
    del topics["/model_odometry"]

if not args.real_world:
    del topics["/start_end"]
def make_transform_stamped(tf_stamped):
    new_tf_stamped = TransformStamped()
    new_tf_stamped.header = tf_stamped.header
    new_tf_stamped.child_frame_id = tf_stamped.child_frame_id
    new_tf_stamped.transform.translation.x = tf_stamped.transform.translation.x
    new_tf_stamped.transform.translation.y = tf_stamped.transform.translation.y
    new_tf_stamped.transform.translation.z = tf_stamped.transform.translation.z
    new_tf_stamped.transform.rotation.x = tf_stamped.transform.rotation.x
    new_tf_stamped.transform.rotation.y = tf_stamped.transform.rotation.y
    new_tf_stamped.transform.rotation.z = tf_stamped.transform.rotation.z
    new_tf_stamped.transform.rotation.w = tf_stamped.transform.rotation.w
    return new_tf_stamped
    
def send_transforms_to_buffer(reader, tf_buffer):
    # tfs need to be handled differently
    transforms_dynamic = []
    transforms_static = []
    connections = [x for x in reader.connections if x.topic == '/tf' or x.topic == '/tf_static']
    for connection, timestamp, raw_data in reader.messages(connections=connections):
        if connection.topic == '/tf':
            transform = reader.deserialize(raw_data, connection.msgtype).transforms
            transforms_dynamic.extend(transform)
        elif connection.topic == '/tf_static':
            transform = reader.deserialize(raw_data, connection.msgtype).transforms
            transforms_static.extend(transform)

    for tf_stamped in transforms_dynamic:
        tf_buffer.set_transform(make_transform_stamped(tf_stamped), 'default_authority')
    for tf_stamped in transforms_static:
        tf_buffer.set_transform_static(make_transform_stamped(tf_stamped), 'default_authority')

def get_tfs(reader, tf_buffer, from_frame, to_frame, time_step_size=8000000):
    time_step = reader.start_time
    data = []
    while time_step < reader.end_time:
        try:
            transform = tf_buffer.lookup_transform(from_frame, to_frame, Time(nanoseconds=time_step))
            data.append([transform.header.stamp.sec + transform.header.stamp.nanosec* 1e-9, transform.transform.translation.x, 
                        transform.transform.translation.y, transform.transform.translation.z, transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z])
        except Exception as e:
            #print(e)
            pass
        time_step += time_step_size # 8ms in nanoseconds
    column_name = f"/tf_{from_frame}_to_{to_frame}_transform"
    df = pd.DataFrame(data, columns=['time', column_name+'.translation.x', column_name+'.translation.y', column_name+'.translation.z', column_name+'.rotation.w', column_name+'.rotation.x', column_name+'.rotation.y', column_name+'.rotation.z'])
    return df

def add_time_stamp(dataframe):
    if "header.stamp.sec" not in dataframe.columns:
        dataframe['time'] = dataframe.index.astype(np.int64) / 10 ** 9
    else:
        dataframe['time'] = dataframe['header.stamp.sec'] + dataframe['header.stamp.nanosec'] * 1e-9
        dataframe = dataframe.drop(['header.stamp.sec', 'header.stamp.nanosec'], axis=1)
    return dataframe

rosbags = [bag for bag in rosbags if bag not in ["counter.pickle", "sampling_space.pickle"]]
bags_to_process = len(rosbags)

rosbags_chunked = np.array_split(rosbags, args.cores)
rosbags_chunked = [list(x) for x in rosbags_chunked if len(x) > 0]

bag_counter = Counter()
lost_bags_counter = Counter()
fall_counter = Counter()
set_lock = Lock()
manager = Manager()
indices_list = manager.list()
def process_bags(rosbags):
    try:
        dataframes = {}
        for bag in rosbags:
            with AnyReader([Path(join(path, bag))]) as reader:
                if len(reader.connections) == 0:
                    print(f"Skipping {bag}, because it is empty.")
                    lost_bags_counter.increment()
                    continue
                set_lock.acquire()
                bag_counter.increment()
                bag_number = bag_counter.value
                set_lock.release()
                # we need to detect the falls and not use the data generated afterwards
                robot_state_df = get_dataframe(reader, "/robot_state", ["state"])
                falling_times = robot_state_df.index[robot_state_df["state"] == 1].tolist()
                first_fall = None
                if len(falling_times) > 0:
                    # convert to seconds without the year
                    first_fall = round(falling_times[0].to_datetime64().astype(int) * 1e-9 % 31536000, 3)
                    print(f"Robot fell at {first_fall} seconds")
                    fall_counter.increment()
                lost_bag = False
                for topic in topics.keys(): 
                    dataframe = get_dataframe(reader, topic, topics[topic])
                    dataframe = add_time_stamp(dataframe)
                    # remove all rows after the first fall
                    if first_fall is not None:
                        dataframe = dataframe.loc[dataframe["time"] < first_fall]
                    if len(dataframe) > 0:
                        dataframe['bag_number'] = bag_number
                    else:
                        print("Empty series")
                        lost_bag = True
                    if topic not in dataframes.keys():
                        dataframes[topic] = dataframe
                    else:
                        dataframes[topic] = pd.concat([dataframes[topic], dataframe])
                    lost_bags_counter.increment() if lost_bag else None
                if args.tf:
                    tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30))
                    send_transforms_to_buffer(reader, tf_buffer)
                    r2l_frame = get_tfs(reader, tf_buffer, 'r_sole', 'l_sole')
                    r2l_frame["bag_number"] = bag_number
                    l2r_frame = get_tfs(reader, tf_buffer, 'l_sole', 'r_sole')
                    l2r_frame["bag_number"] = bag_number
                    right_to_base = get_tfs(reader, tf_buffer, 'r_sole', 'base_link')
                    right_to_base["bag_number"] = bag_number
                    left_to_base = get_tfs(reader, tf_buffer, 'l_sole', 'base_link')
                    left_to_base["bag_number"] = bag_number
                    
                    
                    if "tf_r2l" not in dataframes.keys():
                        dataframes["tf_r2l"] = r2l_frame
                    else:
                        dataframes["tf_r2l"] = pd.concat([dataframes["tf_r2l"], r2l_frame])
                    if "tf_l2r" not in dataframes.keys():
                        dataframes["tf_l2r"] = l2r_frame
                    else:
                        dataframes["tf_l2r"] = pd.concat([dataframes["tf_l2r"], l2r_frame])
                    if "tf_right_to_base" not in dataframes.keys():
                        dataframes["tf_right_to_base"] = right_to_base
                    else:
                        dataframes["tf_right_to_base"] = pd.concat([dataframes["tf_right_to_base"], right_to_base])
                    if "tf_left_to_base" not in dataframes.keys():
                        dataframes["tf_left_to_base"] = left_to_base
                    else:
                        dataframes["tf_left_to_base"] = pd.concat([dataframes["tf_left_to_base"], left_to_base])
            
            print(f"{bag_counter.value}/{bags_to_process} bags processed, currently {bag}")
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        print(f"Error processing {bag}")
        return {}
    
    return dataframes

def process_bags_real_world(rosbags):
    try:
        dataframes = {}

        for bag in rosbags:
            # they are called 02, 03 etc
            bag_number = int(bag.split("_")[0])
            print(f"{bag_counter.value}/{bags_to_process} bags processed, currently {bag}")
            with AnyReader([Path(join(path, bag))]) as reader:
                if len(reader.connections) == 0:
                    print(f"Skipping {bag}, because it is empty.")
                    lost_bags_counter.increment()
                    continue
                set_lock.acquire()
                bag_counter.increment()
                set_lock.release()
                for topic in topics.keys(): 
                    dataframe = get_dataframe(reader, topic, topics[topic])
                    dataframe = add_time_stamp(dataframe)
                    dataframe["bag_number"] = bag_number
                    if topic not in dataframes.keys():
                        dataframes[topic] = dataframe
                    else:
                        dataframes[topic] = pd.concat([dataframes[topic], dataframe])
                if args.tf:
                    tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=1200))
                    send_transforms_to_buffer(reader, tf_buffer)
                    
                    r2l_frame = get_tfs(reader, tf_buffer, 'r_sole', 'l_sole', time_step_size=50000000)
                    l2r_frame = get_tfs(reader, tf_buffer, 'l_sole', 'r_sole', time_step_size=50000000)
                    right_to_base = get_tfs(reader, tf_buffer, 'r_sole', 'base_link', time_step_size=50000000)
                    left_to_base = get_tfs(reader, tf_buffer, 'l_sole', 'base_link', time_step_size=50000000)
                    r2l_frame["bag_number"] = bag_number
                    l2r_frame["bag_number"] = bag_number
                    right_to_base["bag_number"] = bag_number
                    left_to_base["bag_number"] = bag_number

                    if r2l_frame is None or l2r_frame is None:
                        print(f"Error in {bag}")
                        print(f"Empty tf transform")
                        continue
                    if "tf_r2l" not in dataframes.keys():
                        dataframes["tf_r2l"] = r2l_frame
                    else:
                        dataframes["tf_r2l"] = pd.concat([dataframes["tf_r2l"], r2l_frame])
                    if "tf_l2r" not in dataframes.keys():
                        dataframes["tf_l2r"] = l2r_frame
                    else:
                        dataframes["tf_l2r"] = pd.concat([dataframes["tf_l2r"], l2r_frame])
                    if "tf_right_to_base" not in dataframes.keys():
                        dataframes["tf_right_to_base"] = right_to_base
                    else:
                        dataframes["tf_right_to_base"] = pd.concat([dataframes["tf_right_to_base"], right_to_base])
                    if "tf_left_to_base" not in dataframes.keys():
                        dataframes["tf_left_to_base"] = left_to_base
                    else:
                        dataframes["tf_left_to_base"] = pd.concat([dataframes["tf_left_to_base"], left_to_base])
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        print(f"Error processing {bag}")
        return {}
    return dataframes

pool = Pool(min(args.cores, len(rosbags_chunked)))
if args.real_world:
    results = pool.map(process_bags_real_world, rosbags_chunked)
else: 
    results = pool.map(process_bags, rosbags_chunked)
pool.close()
dataframes = {}
for result in results:
    for key in result.keys():
        if key not in dataframes.keys():
            dataframes[key] = result[key]
        else:
            dataframes[key] = pd.concat([dataframes[key], result[key]])

for key in dataframes.keys():
    print(f"Saving {key}")
    dataframes[key] = dataframes[key].sort_values(by=["bag_number", "time"])
    dataframes[key] = dataframes[key].reset_index(drop=True)
    dataframes[key] = dataframes[key][~dataframes[key].index.duplicated(keep='first')]
    file_name = key.replace("/", "__")
    df = dataframes[key]
    df.to_feather(data_path+file_name+ "_rosbag_data.feather")
    if args.csv:
        df.to_csv(data_path+file_name+ "_rosbag_data.csv")
    print(f"path: {data_path}")

print(f"Finished processing {bag_counter.value} bags in {time.time() - Processing_start_time} seconds. We had {lost_bags_counter.value} lost bags.")
print(f"Robot fell {fall_counter.value} times.")
















