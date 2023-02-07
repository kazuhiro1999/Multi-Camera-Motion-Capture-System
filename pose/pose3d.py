import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np


# 3次元姿勢復元
def recover_pose_3d(proj_matrices, keypoints2d_list, th=0.5):
    if len(keypoints2d_list) < 2: # 3D復元には最低2視点が必要
        return None
    proj_matrices = np.array(proj_matrices)
    keypoints2d_list = np.array(keypoints2d_list)
    n_views, n_joints, _ = keypoints2d_list.shape
    assert proj_matrices.shape[0] == n_views

    keypoints3d = []
    for joint_i in range(n_joints):
        points = keypoints2d_list[:,joint_i,:2]
        confidences = keypoints2d_list[:,joint_i,2]
        if np.count_nonzero(confidences > th) > 1:
            alg_confidences = confidences / confidences.sum() + 1e-5
            point_3d = triangulate_points_tf(proj_matrices, points, alg_confidences)
        else:
            point_3d = np.zeros(3)
        keypoints3d.append(point_3d)

    return np.array(keypoints3d)
    
# SVD法
def triangulate_points_tf(proj_matrices, points, confidences):
    n_views = len(proj_matrices)
    
    A = tf.cast(tf.tile(proj_matrices[:, 2:3], (1,2,1)), dtype=tf.float32) * tf.reshape(tf.cast(points, dtype=tf.float32), [n_views, 2, 1])
    A -= tf.cast(proj_matrices[:, :2], dtype=tf.float32)
    A *= tf.reshape(tf.cast(confidences, dtype=tf.float32), [-1, 1, 1])

    u, s, v = tf.linalg.svd(tf.reshape(A, [-1, 4]), full_matrices=False)
    vh = tf.linalg.adjoint(v)

    point_3d_homo = -tf.transpose(vh)[None,:,3]   
    point_3d = tf.transpose(tf.transpose(point_3d_homo)[:-1] / tf.transpose(point_3d_homo)[-1])[0]
    return point_3d

# 再投影
def reprojection(keypoints3d, proj_matrix):
    # keypoints3d : (num_keypoints, 3)
    assert keypoints3d.shape[1] == 3
    num_keypoints = keypoints3d.shape[0]
    points3d = np.vstack((keypoints3d.T, np.ones((1,num_keypoints))))
    keypoints2d = proj_matrix @ points3d
    keypoints2d = keypoints2d[:2,:] / keypoints2d[2,:]
    return keypoints2d.T