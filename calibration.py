import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from camera.setting import CameraSetting
from pose.mp_pose import PoseEstimatorMP
from pose.setting import ModelType


class CameraCalibrator:

    def __init__(self, num_samples=100, min_confidence=0.5):
        self.camera_settings = []
        self.n_cameras = 0
        self.pose_type = ModelType.none
        self.samples = []
        self.num_samples = num_samples
        self.min_confidence = min_confidence
        self.isActive = False

    def start_calibration(self, camera_settings:list, pose_type:ModelType):
        if len(camera_settings) < 2:
            print("At least 2 cameras required to calibrate")
            return
        if pose_type == ModelType.none:
            print("Pose Type must be same")
            return
        self.camera_settings = camera_settings
        self.n_cameras = len(camera_settings)
        self.pose_type = pose_type
        self.isActive = True

    def add_samples(self, keypoints2d_list):
        keypoints2d_list = np.array(keypoints2d_list)
        n_views, n_joints, _ = keypoints2d_list.shape        
        if n_views != self.n_cameras or n_joints != ModelType.get_num_of_joints(self.pose_type):
            return
        self.samples.append(keypoints2d_list)

    def is_sampled(self):
        return len(self.samples) > self.num_samples
    
    def calibrate_cameras(self, base_i=0, pair_i=1):
        keypoints2d_list = np.array(self.samples)
        n_frames, n_views, n_joints, _ = keypoints2d_list.shape
        sample_points = []
        for frame_i in range(n_frames):
            for joint_i in range(n_joints):
                points = keypoints2d_list[frame_i,:,joint_i]
                if np.all(points[:,2] < self.min_confidence): # すべてのカメラで検出した点のみ使用
                    sample_points.append(points[:,:2])
        sample_points = np.array(sample_points).transpose([1,0,2]) # shape:(n_views, n_points, 2)
        points3d = calibrate_cameras(self.camera_settings, sample_points, base_i, pair_i)
        return

    def calibrate_room(self):
        # 現在のカメラパラメータで3次元推定
        keypoints3d_list = []

        calibrate_room(self.camera_settings, keypoints3d_list, self.pose_type)
        return

def estimate_initial_extrinsic(pts1, pts2, K):
    # pts : (N,2)
    E, mask = cv2.findEssentialMat(pts2, pts1, K, method=cv2.FM_LMEDS)
    pts, R, t, mask = cv2.recoverPose(E, pts2, pts1, K)
    return R, t

def calibrate_cameras(camera_setting_list, keypoints2d_list, base_i=0, pair_i=1):
    n_views, n_points, _ = keypoints2d_list.shape

    pts1 = keypoints2d_list[base_i].reshape([-1,2])
    pts2 = keypoints2d_list[pair_i].reshape([-1,2])
    K = camera_setting_list[base_i].get_camera_matrix()
    R, t = estimate_initial_extrinsic(pts1, pts2, K)
    camera_setting_list[base_i].set_transform(position=CameraSetting.DefaultPosition, rotation=CameraSetting.DefaultRotation)
    camera_setting_list[pair_i].set_transform(position=t, rotation=R)

    points3d = cv2.triangulatePoints(
        camera_setting_list[base_i].proj_matrix,
        camera_setting_list[pair_i].proj_matrix,
        pts1.T,
        pts2.T
    )
    points3d = (points3d[:3,:] / points3d[3,:]).T

    for view_i in range(n_views):
        if view_i == base_i or view_i == pair_i:
            continue
        pts = keypoints2d_list[view_i].reshape([-1,2]).astype(np.float32)
        ret, rc, tc, mask = cv2.solvePnPRansac(points3d, pts, K, np.zeros([4,1]))
        Rc = cv2.Rodrigues(rc)[0]
        R = Rc.T
        t = -Rc.T @ tc
        camera_setting_list[view_i].set_transform(position=t, rotation=R)

    return points3d


def calibrate_room(camera_setting_list, keypoints3d_list, pose_type:ModelType):
    o_pos = determine_center_position(keypoints3d_list, pose_type)
    o_mat = determine_forward_rotation(keypoints3d_list, pose_type)
    scale = determine_scale(keypoints3d_list, pose_type, Height=1.6, )
    o_rot = Rotation.from_matrix(o_mat)

    for camera_setting in camera_setting_list:
        t = o_rot.apply(camera_setting.position - o_pos).reshape([3,1]) * scale
        R = (o_rot * Rotation.from_euler('zyx', camera_setting.rotation)).as_matrix()
        camera_setting.set_transform(t, R)


def determine_center_position(keypoints3d_list, pose_type:ModelType):
    # keypoints3d_list : (n_frames, n_joints, 3)
    if pose_type == ModelType.MediapipePose or pose_type == ModelType.MediapipeHolistic:
        l_foot = keypoints3d_list[:, PoseEstimatorMP.KEYPOINT_DICT['left_ankle']]
        r_foot = keypoints3d_list[:, PoseEstimatorMP.KEYPOINT_DICT['right_ankle']]
        m_foot = (l_foot + r_foot) / 2
        center_position = m_foot.mean(axis=0)
    else:
        center_position = CameraSetting.DefaultPosition
    return center_position


def determine_forward_rotation(keypoints3d_list, pose_type:ModelType):
    # keypoints3d_list : (n_frames, n_joints, 3)
    if pose_type == ModelType.MediapipePose or pose_type == ModelType.MediapipeHolistic:
        l_shoulder = keypoints3d_list[:, PoseEstimatorMP.KEYPOINT_DICT['left_shoulder']]
        r_shoulder = keypoints3d_list[:, PoseEstimatorMP.KEYPOINT_DICT['right_shoulder']]
        l_hips = keypoints3d_list[:, PoseEstimatorMP.KEYPOINT_DICT['left_hip']]
        r_hips = keypoints3d_list[:, PoseEstimatorMP.KEYPOINT_DICT['right_hip']]
        m_hips = (l_hips + r_hips) / 2
        forward_vector = np.cross(l_shoulder - m_hips, r_shoulder - m_hips).mean(axis=0)
        forward_vector = forward_vector / np.linalg.norm(forward_vector)
        rotation_matrix = (forward_vector.reshape([3,1]) @ CameraSetting.Forward.reshape([1,3])).astype(np.float32)
    else:
        rotation_matrix = CameraSetting.DefaultRotation
    return rotation_matrix

def determine_scale(keypoints3d_list, pose_type:ModelType, Height=1.0):
    if pose_type == ModelType.MediapipePose or pose_type == ModelType.MediapipeHolistic:
        head = keypoints3d_list[:, PoseEstimatorMP.KEYPOINT_DICT['nose']]
        l_foot = keypoints3d_list[:, PoseEstimatorMP.KEYPOINT_DICT['left_ankle']]
        r_foot = keypoints3d_list[:, PoseEstimatorMP.KEYPOINT_DICT['right_ankle']]
        m_foot = (l_foot + r_foot) / 2
        height = np.linalg.norm(head - m_foot, axis=-1).mean()
    else:
        height = 1.0
    scale = Height / height
    return scale

