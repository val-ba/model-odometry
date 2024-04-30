import torch
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from biped_interfaces.msg import Phase
from geometry_msgs.msg import TransformStamped
import tf2_ros as tf2
from rclpy.duration import Duration
from nav_msgs.msg import Odometry
import numpy as np
import transforms3d as tf3d
from transforms3d.euler import quat2euler
from rclpy.time import Time
from rclpy import Parameter
from collections import deque
from ros2_numpy import numpify
from bitbots_tf_listener import TransformListener
from scripts.normalize_sample import normalize, denormalize
from Model.load_model import load_model_from_name_string

class MessageBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size) 

    def callback(self, msg):
        timestamp = Time.from_msg(msg.header.stamp)
        self.buffer.append((timestamp, msg))

    def get_closest_message(self, target_time):
        target_time = Time.from_msg(target_time)
        closest_msg = min(self.buffer, key=lambda x: abs((x[0] - target_time).nanoseconds))
        # pop all the messages which are older
        while self.buffer[0][0] < closest_msg[0]:
            self.buffer.popleft()
        return closest_msg[1]
    
class ModelOdometry(Node):
    def __init__(self) -> None:
        """
        Node that uses a model to calculate the odometry.
        """
        
        super().__init__("model_odometry", automatically_declare_parameters_from_overrides=True)
        self._package_path = get_package_share_directory('odometry_misc')
        
        parameter = Parameter("use_sim_time", Parameter.Type.BOOL, True) # if in simulation
        self.set_parameters([parameter])
        
        model_name = "" # needs to be set
        self.model, self.architecture = load_model_from_name_string(model_name)
        if self.architecture == "LSTM":
            self.hidden_layer, self.gates = self.model.init_hidden(1)
        elif self.architecture == "RNN":
            self.hidden_layer = self.model.init_hidden(1)

        self.tf_buffer = tf2.Buffer(cache_time=Duration(seconds=30))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.br = tf2.TransformBroadcaster(self)

        self.imu_buffer = MessageBuffer(1000)
        self.walk_support_state_buffer = MessageBuffer(100)

        self.imu_sub = self.create_subscription(Imu, "/imu/data", self.imu_buffer.callback, 1)
        self.walk_support_state_sub = self.create_subscription(Phase, "/foot_pressure/walk_support_state", self.walk_support_state_buffer.callback, 1)
        self.walk_support_state_sub = self.create_subscription(Phase, "/walk_support_state", self.walk_support_state_buffer.callback, 1) # may be changed to foot pressure wsp

        self.base_link_frame = "base_link"
        self.odom_frame = "odom"

        self.odom_pub = self.create_publisher(Odometry, "/model_odometry", 1)

        self.odometry_to_support_sole = tf3d.affines.compose([0, 0, 0], np.eye(3), np.ones(3)) 

        self.last_phase = 1
        self.current_phase = 0
        self.double_support_time_stamp = self.get_clock().now().to_msg()

    def get_tfs(self, time_stamp):
        try:
            return self.tf_buffer.lookup_transform("l_sole", "r_sole", time_stamp)
        except Exception as e:
            print(e)
    
    def get_prediction(self, model_input):
        if self.architecture == "LSTM":
            pred, (self.hidden_layer, self.gates) = self.model(torch.tensor(model_input).float(), (self.hidden_layer, self.gates))
        elif self.architecture == "MLP":
            pred = self.model(torch.tensor(model_input).float())
        elif self.architecture == "RNN":
            pred, self.hidden_layer = self.model(torch.tensor(model_input).float(), self.hidden_layer)
        x, y, yaw = denormalize(pred.detach().numpy().flatten())
        return x, y, yaw
    
    def prepare_model_input(self, imu, l2r, phase):
        imu_data = [ imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z,imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z, imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w]
        l2r_euler = quat2euler([l2r.transform.rotation.w, l2r.transform.rotation.x, l2r.transform.rotation.y, l2r.transform.rotation.z])
        l2r_xy_yaw = [l2r.transform.translation.x, l2r.transform.translation.y, l2r_euler[2]]
        # if phase is 0, then -1
        if phase == 0:
            phase = 1
        elif phase == 1:
            phase = -1
        return normalize(imu_data + l2r_xy_yaw + [phase])
    
    def loop(self):
        time_stamp = self.get_clock().now().to_msg()
        if len(self.walk_support_state_buffer.buffer) > 0 and len(self.imu_buffer.buffer) > 0:
            # get the latest message
            walk_support_state_msg = self.walk_support_state_buffer.buffer.popleft()[1] # left means first element added
            if walk_support_state_msg.phase == 2:
                self.double_support_time_stamp = walk_support_state_msg.header.stamp
            else:
                self.current_phase = walk_support_state_msg.phase
            
            if self.current_phase != self.last_phase:
                ds_time_stamp = self.double_support_time_stamp
                print(walk_support_state_msg.phase, self.last_phase)

                # get input
                imu_msg = self.imu_buffer.get_closest_message(ds_time_stamp)
                l2r= self.get_tfs(ds_time_stamp)
                model_input = self.prepare_model_input(imu_msg, l2r, self.last_phase)
                x, y, yaw = self.get_prediction(model_input)

                previous_to_current_sole = tf3d.affines.compose([x, y, 0], tf3d.euler.euler2mat(0, 0, yaw), np.ones(3))
                self.odometry_to_support_sole = self.odometry_to_support_sole @ previous_to_current_sole
                self.last_phase = self.current_phase
                
        curr_support_link = "r_sole"
        if self.last_phase == 0:
            curr_support_link = "l_sole"
        try:
            support_sole_to_base_link = self.tf_buffer.lookup_transform(curr_support_link, self.base_link_frame, time_stamp, Duration(seconds=1.0))
            support_sole_to_base_link = numpify(support_sole_to_base_link.transform)
            odom_to_base_link = self.odometry_to_support_sole @ support_sole_to_base_link
            translation, rotation, _, _ = tf3d.affines.decompose44(odom_to_base_link)

            # create odometry message
            odom_msg = TransformStamped()
            odom_msg.header.stamp = time_stamp
            odom_msg.header.frame_id = self.odom_frame
            odom_msg.child_frame_id = self.base_link_frame
            # set the transform
            odom_msg.transform.translation.x = translation[0]
            odom_msg.transform.translation.y = translation[1]
            odom_msg.transform.translation.z = translation[2]
            
            rotation = tf3d.quaternions.mat2quat(rotation)
            odom_msg.transform.rotation.x = rotation[1]
            odom_msg.transform.rotation.y = rotation[2]
            odom_msg.transform.rotation.z = rotation[3]
            odom_msg.transform.rotation.w = rotation[0]
            self.br.sendTransform(odom_msg)
            # create odometry message
            odom_msg = Odometry()
            odom_msg.header.stamp = time_stamp
            odom_msg.header.frame_id = self.odom_frame
            odom_msg.child_frame_id = self.base_link_frame
            # set the transform
            odom_msg.pose.pose.position.x = translation[0]
            odom_msg.pose.pose.position.y = translation[1]
            odom_msg.pose.pose.position.z = translation[2]
            odom_msg.pose.pose.orientation.x = rotation[1]
            odom_msg.pose.pose.orientation.y = rotation[2]
            odom_msg.pose.pose.orientation.z = rotation[3]
            odom_msg.pose.pose.orientation.w = rotation[0]
            self.odom_pub.publish(odom_msg)
        except Exception as e:
            print(e)

def main(args=None):
    rclpy.init(args=args)
    model_odometry = ModelOdometry()
    timer_period = 0.005  # seconds
    try:
        model_odometry.create_timer(timer_period, model_odometry.loop)
        rclpy.spin(model_odometry)
    except KeyboardInterrupt:
        model_odometry.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()