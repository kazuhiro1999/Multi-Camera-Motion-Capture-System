'''
Mediapipe Holistic Model for 2D pose estimation from image
please take care about the key-points are not same 
'''


import cv2
from tools.visualization import draw_keypoints
import numpy as np
import mediapipe as mp

from pose.setting import ModelType


mp_holistic = mp.solutions.holistic

class MediapipeHolistic:
    KEYPOINT_DICT = {
        'nose': 0,
        'left_inner_eye': 1,
        'left_eye': 2,
        'left_outer_eye': 3,
        'right_inner_eye': 4,
        'right_eye': 5,
        'right_outer_eye': 6,
        'left_ear': 7,
        'right_ear':8,
        'left_mouth': 9,
        'right_mouth': 10,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_outer_hand': 17,
        'right_outer_hand': 18,
        'left_hand_tip': 19,
        'right_hand_tip': 20,
        'left_inner_hand': 21,
        'right_inner_hand': 22,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28,
        'left_heel': 29,
        'right_heel': 30,
        'left_toe': 31,
        'right_toe': 32,

        'left_wrist_1': 33,
        'left_wrist_2': 34,
        'left_thumb_base': 35,
        'left_thumb_1': 36,
        'left_thumb_end': 37,
        'left_index_base': 38,
        'left_index_2': 39,
        'left_index_1': 40,
        'left_index_end': 41,
        'left_middle_base': 42,
        'left_middle_2': 43,
        'left_middle_1': 44,
        'left_middle_end': 45,
        'left_ring_base': 46,
        'left_ring_2': 47,
        'left_ring_1': 48,
        'left_ring_end': 49,
        'left_little_base':50,
        'left_little_2':51,
        'left_little_1':52,
        'left_little_end':53,

        'right_wrist_1': 54,
        'right_wrist_2': 55,
        'right_thumb_base': 56,
        'right_thumb_1': 57,
        'right_thumb_end': 58,
        'right_index_base': 59,
        'right_index_2': 60,
        'right_index_1': 61,
        'right_index_end': 62,
        'right_middle_base': 63,
        'right_middle_2': 64,
        'right_middle_1': 65,
        'right_middle_end': 66,
        'right_ring_base': 67,
        'right_ring_2': 68,
        'right_ring_1': 69,
        'right_ring_end': 70,
        'right_little_base':71,
        'right_little_2':72,
        'right_little_1':73,
        'right_little_end':74,
    }

    def __init__(self):
        self.type = ModelType.MediapipeHolistic
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, image):
        image_height, image_width = image.shape[:2]
        results = self.holistic.process(image)

        if results.pose_landmarks is not None:
            pose_keypoints = self.landmarks_to_keypoints2d(results.pose_landmarks, (image_width, image_height), type='pose')
        else:
            return None
        
        if results.left_hand_landmarks is not None:
            left_hand_keypoints = self.landmarks_to_keypoints2d(results.left_hand_landmarks, (image_width, image_height), type='left_hand')
        else:
            left_hand_keypoints = np.zeros([21,3])

        if results.right_hand_landmarks is not None:
            right_hand_keypoints = self.landmarks_to_keypoints2d(results.right_hand_landmarks, (image_width, image_height), type='right_hand')            
        else:
            right_hand_keypoints = np.zeros([21,3])

        keypoints2d = np.concatenate([pose_keypoints, left_hand_keypoints, right_hand_keypoints], axis=0)
        return keypoints2d

    def landmarks_to_keypoints2d(self, landmarks, image_shape, type):
        image_width, image_height = image_shape
        keypoints2d = []
        for landmark in landmarks.landmark:
            x = landmark.x * image_width
            y = landmark.y * image_height
            if type == 'pose':
                keypoints2d.append([x, y, landmark.visibility])
            else:
                keypoints2d.append([x, y, landmark.visibility + 1])
        return np.array(keypoints2d)


if __name__ == '__main__':
    from camera.camera import USB_Camera
    
    camera = USB_Camera('cam 0', device_id=0)
    camera.open()

    pose_estimator = MediapipeHolistic()

    while True:
        img = camera.get_image()
        if img is None:
            continue

        keypoints2d = pose_estimator.process(img)

        debug_image = draw_keypoints(img, keypoints2d)
        cv2.imshow('image', debug_image)
         
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()