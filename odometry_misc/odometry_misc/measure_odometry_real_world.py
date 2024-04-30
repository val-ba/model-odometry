import rclpy
from rclpy.node import Node
import subprocess
from geometry_msgs.msg import Twist
import numpy.random as random
import numpy as np
from rclpy.duration import Duration
from std_srvs.srv import Empty
import pickle
import os
import sys
from bitbots_msgs.msg import Buttons
from std_msgs.msg import Int16

current_directory = os.getcwd()
counter_dict = {"velocity_space_counter": 0, "transition_counter": 0, "random": False}
sampling_space_dict = {"sampling_space": None, "transitions": None}
# check whether counter.pickle and sampling_space.pickle is in current directory
if not os.path.isfile(os.path.join(current_directory, "counter.pickle")) or not os.path.isfile(os.path.join(current_directory, "sampling_space.pickle")):
    sampling_method = input("Need to create a sampling space: Choose 1 for random walk, 2 for grid sampling: ")
    if sampling_method != "1" and sampling_method != "2":
        print("Invalid input, exiting.")
        sys.exit()


    sampling_space = []
    intervals = {"x": (-0.05, 0.1), "y": (-0.04, 0.04), "yaw": (-0.3, 0.3)} 
    transitions = []
    max_trans_vel_forward = 0.1
    max_trans_vel_backward = 0.05
    if sampling_method == "1":
        counter_dict["random"] = True
        for i in range(0, 10000):
            sampeld_x = random.uniform(intervals["x"][0], intervals["x"][1])
            if sampeld_x > 0:
                max_y_vel = max_trans_vel_forward - sampeld_x
                y_sampeld = random.uniform(-max_y_vel, max_y_vel)
            else:
                max_y_vel = max_trans_vel_backward - sampeld_x
                y_sampeld = random.uniform(-max_y_vel, max_y_vel)


            sampling_space.append((sampeld_x, y_sampeld, random.uniform(intervals["yaw"][0], intervals["yaw"][1])))
        for i in range(0, 10000):
            transitions.append((random.uniform(-0.0125, 0.025), random.uniform(-0.01, 0.01), random.uniform(-0.1, 0.1))) 

    elif sampling_method == "2":
        sampling_rate = 10 
        x_sampling_space = np.linspace(intervals["x"][0], intervals["x"][1], num=sampling_rate)
        y_sampling_space = np.linspace(intervals["y"][0], intervals["y"][1], num=sampling_rate)
        yaw_sampling_space = np.linspace(intervals["yaw"][0], intervals["yaw"][1], num=sampling_rate)

        for i in range(0, sampling_rate):
            for j in range(0, sampling_rate):
                for k in range(0, sampling_rate):
                    sampling_space.append((x_sampling_space[i], y_sampling_space[j], yaw_sampling_space[k]))

        for i in [0.025, 0.05]:
            transitions.append((i, 0, 0))
            transitions.append((-i, 0, 0))
        for i in [0.015, -0.015]:
            transitions.append((0, i, 0))
        for i in [0.2, -0.2, 0.1, -0.1]:
            transitions.append((0, 0, i))

    sampling_space_dict = {"sampling_space": sampling_space, "transitions": transitions}
    with open("sampling_space.pickle", "wb") as f:
        pickle.dump(sampling_space_dict, f)
    with open("counter.pickle", "wb") as f:
        pickle.dump(counter_dict, f)
else:
    print("Found counter.pickle and sampling_space.pickle, loading...")
    with open("counter.pickle", "rb") as f:
        counter_dict = pickle.load(f)
    with open("sampling_space.pickle", "rb") as f:
        sampling_space_dict = pickle.load(f)
    print("Done loading.")
    print(f"Current counter: {counter_dict}")
    print(f" The method of sampling was {'random walk' if counter_dict['random'] else 'grid sampling'}")
    input("Press enter to continue...")
run_number = int(input("Enter the run number: "))
print(f"Starting run {run_number}.")

class WalkingNode(Node):
    """Creates a node which lets the robot run through a set of velocities and transitions applied to them.
    This is the counter part to the WalkingNode in measure_odometry.py for the real world.

    """
    def __init__(self):
        # create node
        super().__init__("WalkingNode")
        self.pub = self.create_publisher(Twist, 'cmd_vel', 1)
        self.client = self.create_client(Empty, "/reset_pose")
        self.start_end_pub = self.create_publisher(Int16, "/start_end", 10)
        self.button_sub = self.create_subscription(Buttons, "/buttons", self.button_callback, 10)
        self.client.wait_for_service(3.0)
        self.twist = Twist()
        self.create_timer(1, self.loop) 
       
        self.velocity_sample_counter = counter_dict["velocity_space_counter"]
        self.sample_transition_counter = counter_dict["transition_counter"]
        self.sampling_space = sampling_space_dict["sampling_space"]
        self.transitions = sampling_space_dict["transitions"]
        self.random = counter_dict["random"]
        self.topics = " /start_end /foot_pressure/walk_support_state /animation /attached_collision_object /clock /cmd_vel /collision_object /cop_l /cop_r /core/power_switch_status /debug/dsd/hcm /debug/dynamic_kick/debug /debug/dynamic_kick/flying_foot_spline /debug/dynamic_kick/kick_windup_point /debug/dynamic_kick/received_goal /debug/dynamic_kick/trunk_spline /debug/dynup/received_goal /debug_markers /diagnostics_agg /display_contacts /display_planned_path /dynamic_kick_support_state /dynup_engine_debug /dynup_engine_marker /dynup_motor_goals /foot_pressure_left/filtered /foot_pressure_left/raw /foot_pressure_right/filtered /foot_pressure_right/raw /hcm_deactivate /head_motor_goals /imu/data /imu/data_raw /imu_head/data /imu_head/data_raw /imu_l_foot/data /imu_l_foot/data_raw /imu_r_foot/data /imu_r_foot/data_raw /joint_states /kick /kick_motor_goals /model_states /motion_odometry /motion_plan_request /parameter_events /pause /pid_state /planning_scene /planning_scene_world /record_motor_goals /robot_description /robot_description_semantic /robot_state /rosout /speak /step /tf /tf_static /trajectory_execution_event /walk_debug /walk_debug_marker /walk_engine_debug /walk_engine_odometry /walk_support_state /walking_motor_goals /camera/image_raw /foot_pressure/summed_pressure_left /foot_pressure/summed_pressure_right"
        self.current_bag = None
        self.running = False
        self.phase = 1
        self.start_time = self.get_clock().now()


        self.button_pressed = False
        self.trial_counter = 0

        unique_name = f"{run_number:02d}"
        self.current_bag = subprocess.Popen(["ros2", "bag", "record", "-o", unique_name, *self.topics.split()]) 
        print("Lift the robot high once to later sync the data.")

        print("Then press button 2 to start recording.")

    def button_callback(self, msg):
        if msg.button2 == True:
            self.button_pressed = True
            


    def walk(self, x, y, yaw):
        self.twist.angular.z = yaw
        self.twist.linear.x = x
        self.twist.linear.y = y
        self.twist.angular.x = 0.0

    def stop_walk(self):    
        self.twist.linear.x = 0.0
        self.twist.linear.y = 0.0
        self.twist.angular.z = 0.0 
        self.twist.angular.x = -1.0

    def loop(self): 
        if self.phase == 1 and self.button_pressed == True:
            self.start_time = self.get_clock().now()
            self.button_pressed = False
            self.start_end_pub.publish(Int16(data=1))
            
            print(f"First velocity: X: {self.sampling_space[self.velocity_sample_counter][0]}, Y: {self.sampling_space[self.velocity_sample_counter][1]}, Yaw: {self.sampling_space[self.velocity_sample_counter][2]}")
            print(f"Second velocity: X: {self.transitions[self.sample_transition_counter][0]}, Y: {self.transitions[self.sample_transition_counter][1]}, Yaw: {self.transitions[self.sample_transition_counter][2]}")
            self.walk(* self.sampling_space[self.velocity_sample_counter])
            self.pub.publish(self.twist)
            self.phase = 2 

        elif self.phase == 2 and self.get_clock().now() - self.start_time > Duration(seconds=5):
            self.walk(*[sum(x) for x in zip( self.sampling_space[self.velocity_sample_counter],  self.transitions[self.velocity_sample_counter])])
            self.pub.publish(self.twist)
            self.phase = 3 

        elif self.phase == 3 and self.get_clock().now() - self.start_time > Duration(seconds=10):
            self.stop_walk()
            self.pub.publish(self.twist)
            self.start_end_pub.publish(Int16(data=0))
            self.phase = 4

        elif self.phase == 4:
            self.velocity_sample_counter += 1
            # save the current counter to a pickle file
            counter_dict["velocity_space_counter"] = self.velocity_sample_counter
            counter_dict["transition_counter"] = self.sample_transition_counter
            with open("counter.pickle", "wb") as f:
                pickle.dump(counter_dict, f)
            if self.velocity_sample_counter > len(self.sampling_space):
                raise SystemExit
            self.phase = 1
            self.trial_counter += 1
            print(f"Done with sample {self.velocity_sample_counter} and transition {self.sample_transition_counter}")
            print("Press button 2 to continue after placing the robot.")

     
def main(args=None):
    rclpy.init(args=args)
    node = WalkingNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        node.current_bag.terminate()
        node.destroy_node()
        rclpy.shutdown()

    node.destroy_node()
if __name__ == '__main__':
    main()
