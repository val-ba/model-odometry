import transforms3d as t3d
import numpy as np
import matplotlib.pyplot as plt
from transforms3d.quaternions import quat2mat
from transforms3d.affines import compose

def get_matrix_from_xyyaw(x, y, yaw) -> np.ndarray:
    """
    Get a 4x4 matrix from x, y, yaw
    """
    T = t3d.affines.compose([x, y, 0], t3d.euler.euler2mat(0, 0, yaw), [1, 1, 1])
    return T
def get_xyyaw_from_matrix(T: np.ndarray) -> np.ndarray:
    """
    Get x, y, yaw from a 4x4 matrix
    """
    translation, rotation, scale, _ = t3d.affines.decompose44(T)
    x, y, _ = translation
    yaw = t3d.euler.mat2euler(rotation)[2]
    return np.array([x, y, yaw])

def get_transforms(world_to_a_foot, world_to_b_foot):
    '''Computes the translation and rotation from absolute world coordinates to a foot and b foot'''
    rotations = []
    translations = []
    for world_to_a_foot_single, world_to_b_foot_single in zip(world_to_a_foot, world_to_b_foot):
        # wa ^ -1  * wb
        a_to_b = np.linalg.inv(world_to_a_foot_single) @ world_to_b_foot_single
        xyyaw = get_xyyaw_from_matrix(a_to_b)
        rotations.append(xyyaw[2])
        translations.append(xyyaw[:2])
    return np.array(translations), np.array(rotations)

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

def transforms_to_world_frame(poses):
    """
    The world frame is assumed to be at 0, 0, 0 with no rotation.
    We assume that the transform from local to world frame is the inverse of the first pose.
    Input: poses: poses specified as matrixes in the local frame
    
    """
    poses_world = []
    T_world = np.linalg.inv(poses[0])
    for pose in poses:
        poses_world.append(np.dot(T_world, pose))
    return poses_world

def get_points_from_matrix(T: np.ndarray):
    """
    Get the points of the matrix
    """
    return T[0:3, 3]

if __name__ == "__main__":
    poses = []
    number_of_poses = 3
    for i in range(number_of_poses):
        # we create some random transforms to test the functions
        T = np.eye(4)
        T[0:3, 0:3] = t3d.euler.euler2mat(np.random.rand(), np.random.rand(), np.random.rand())
        T[0:3, 3] = np.random.rand(3) 
        
        poses.append(T)
    poses_world = transforms_to_world_frame(poses)
    assert np.allclose(poses_world[0], np.eye(4)), "The first pose should be identity in the world frame"
    for i in range(1, number_of_poses):
        T_w = np.dot(np.linalg.inv(poses_world[i-1]), poses_world[i])
        T_p = np.dot(np.linalg.inv(poses[i-1]), poses[i])
        assert np.allclose(T_w, T_p), "The transform between poses is not correct. It should be relatively the same."
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for pose in poses_world:
        ax.scatter(pose[0, 3], pose[1, 3], pose[2, 3], color='r')
        for i in range(3):
            ax.quiver(pose[0, 3], pose[1, 3], pose[2, 3], pose[0, i], pose[1, i], pose[2, i], color='r')
    for pose in poses:
        ax.scatter(pose[0, 3], pose[1, 3], pose[2, 3], color='b')
        for i in range(3):
            ax.quiver(pose[0, 3], pose[1, 3], pose[2, 3], pose[0, i], pose[1, i], pose[2, i], color='b')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1,1,1])
    
    
    plt.show()
    x, y, yaw = 1, 2, np.pi/2
    T = get_matrix_from_xyyaw(x, y, yaw)
    assert np.allclose(T[0:2, 3], [x, y]), "The x and y values are not correct"
    assert np.allclose(t3d.euler.mat2euler(T[0:3, 0:3]), [0, 0, yaw]), "The yaw value is not correct"
    

    print("All tests passed!")