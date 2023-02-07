import cv2
import numpy as np
from scipy.spatial.transform import Rotation


class SpaceCalibrator:
    def __init__(self, cameras, num_samples=100):
        self.isActive = False
        self.num_samples = num_samples
        self.n_views = len(cameras)
        self.camera_settings = [camera.camera_setting for camera in cameras]
        self.sample_points = []

    def add_points(self, keypoints2d_list, th=0.5):
        assert keypoints2d_list.shape[0] == self.n_views

        # 全ての視点で全ての関節点が見えている場合のみ追加
        if keypoints2d_list[:,:,2].min() > th:
            self.sample_points.append(keypoints2d_list)
            return True
        else:
            return False

    def calibrate(self):
        samples = np.array(self.sample_points) # (n_frames, n_views, n_joints, 3)
        camera_settings, keypoints3d_list = calibrate_cameras(self.camera_settings, samples)
        room_calibration(camera_settings, keypoints3d_list)
        

def estimate_initial_extrinsic(pts1, pts2, K):
    # pts : (N,2)
    E, mask = cv2.findEssentialMat(pts2, pts1, K, method=cv2.FM_LMEDS)
    pts, R, t, mask = cv2.recoverPose(E, pts2, pts1, K)
    return R, t

def calibrate_cameras(camera_setting_list, keypoints2d_list, base_i=0, pair_i=1):
    n_frames, n_views, n_joints = keypoints2d_list.shape[:3]

    pts1 = keypoints2d_list[:,base_i,:,:2].reshape([-1,2])
    pts2 = keypoints2d_list[:,pair_i,:,:2].reshape([-1,2])
    K = camera_setting_list[base_i].camera_matrix
    R, t = estimate_initial_extrinsic(pts1, pts2, K)
    camera_setting_list[base_i].set_transform(np.zeros([3,1]), np.eye(3,3))
    camera_setting_list[pair_i].set_transform(t, R)

    points3d = cv2.triangulatePoints(
        camera_setting_list[base_i].proj_matrix,
        camera_setting_list[pair_i].proj_matrix,
        pts1.T,
        pts2.T
    )
    points3d = (points3d[:3,:] / points3d[3,:]).T

    keypoints3d_list = points3d.reshape([n_frames, n_joints, 3])

    for view_i in range(n_views):
        if view_i == base_i or view_i == pair_i:
            continue
        pts = keypoints2d_list[:,view_i,:,:2].reshape([-1,2]).astype(np.float32)
        ret, rc, tc, mask = cv2.solvePnPRansac(points3d, pts, K, np.zeros([4,1]))
        Rc = cv2.Rodrigues(rc)[0]
        R = Rc.T
        t = -Rc.T @ tc
        camera_setting_list[view_i].set_transform(t, R)

    return camera_setting_list, keypoints3d_list


def room_calibration(camera_setting_list, keypoints3d_list):
    o_pos = determine_center_position(keypoints3d_list)
    o_mat = determine_forward_rotation(keypoints3d_list)
    scale = determine_scale(keypoints3d_list, Height=1.7)
    o_rot = Rotation.from_matrix(o_mat)

    for camera_setting in camera_setting_list:
        t = o_rot.apply(camera_setting.position - o_pos).reshape([3,1]) * scale
        R = (o_rot * Rotation.from_euler('zyx', camera_setting.rotation)).as_matrix()
        camera_setting.set_transform(t, R)


def determine_center_position(keypoints3d_list):
    # keypoints3d_list : (n_frames, n_joints, 3)
    l_foot = keypoints3d_list[:, KEYPOINT_DICT['left_ankle']]
    r_foot = keypoints3d_list[:, KEYPOINT_DICT['right_ankle']]
    m_foot = (l_foot + r_foot) / 2
    center_position = m_foot.mean(axis=0)
    return center_position


def determine_forward_rotation(keypoints3d_list):
    # keypoints3d_list : (n_frames, n_joints, 3)
    l_shoulder = keypoints3d_list[:, KEYPOINT_DICT['left_shoulder']]
    r_shoulder = keypoints3d_list[:, KEYPOINT_DICT['right_shoulder']]
    l_hips = keypoints3d_list[:, KEYPOINT_DICT['left_hip']]
    r_hips = keypoints3d_list[:, KEYPOINT_DICT['right_hip']]
    m_hips = (l_hips + r_hips) / 2
    forward_vector = np.cross(l_shoulder - m_hips, r_shoulder - m_hips).mean(axis=0)
    forward_vector = forward_vector / np.linalg.norm(forward_vector)
    forward_global = np.array([0,0,1])
    rotation_matrix = (forward_vector.reshape([3,1]) @ forward_global.reshape([1,3])).astype(np.float32)
    return rotation_matrix

def determine_scale(keypoints3d_list, Height=1.0):
    head = keypoints3d_list[:, KEYPOINT_DICT['nose']]
    l_foot = keypoints3d_list[:, KEYPOINT_DICT['left_ankle']]
    r_foot = keypoints3d_list[:, KEYPOINT_DICT['right_ankle']]
    m_foot = (l_foot + r_foot) / 2
    height = np.linalg.norm(head - m_foot, axis=-1).mean()
    scale = Height / height
    return scale