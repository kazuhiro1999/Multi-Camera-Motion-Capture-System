import cv2 as cv
import cv2
import numpy as np
import mediapipe as mp

from pose.setting import ModelType


mp_pose = mp.solutions.pose

class PoseEstimatorMP:
    KEYPOINT_DICT = {
        'nose': 0,
        'right_inner_eye': 1,
        'right_eye': 1,
        'right_outer_eye': 3,
        'left_inner_eye': 4,
        'left_eye': 5,
        'left_outer_eye': 6,
        'right_ear': 7,
        'left_ear':8,
        'right_mouth': 9,
        'left_mouth': 10,
        'right_shoulder': 11,
        'left_shoulder': 12,
        'right_elbow': 13,
        'left_elbow': 14,
        'right_wrist': 15,
        'left_wrist': 16,
        'right_outer_hand': 17,
        'left_outer_hand': 18,
        'right_hand_tip': 19,
        'left_hand_tip': 20,
        'right_inner_hand': 21,
        'left_iiner_hand': 22,
        'right_hip': 23,
        'left_hip': 24,
        'right_knee': 25,
        'left_knee': 26,
        'right_ankle': 27,
        'left_ankle': 28,
        'right_heel': 29,
        'left_heel': 30,
        'right_toe': 31,
        'left_toe': 32
    }

    def __init__(self):
        self.type = ModelType.Mediapipe
        self.pose = mp_pose.Pose(
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, image):
        image_height, image_width = image.shape[:2]
        results = self.pose.process(image)
        if results.pose_landmarks is not None:
            keypoints2d = self.landmarks_to_keypoints2d(results.pose_landmarks, (image_width, image_height))
        else:
            keypoints2d = None
        return keypoints2d

    def landmarks_to_keypoints2d(self, landmarks, image_shape):
        image_width, image_height = image_shape
        keypoints2d = []
        for landmark in landmarks.landmark:
            x = landmark.x * image_width
            y = landmark.y * image_height
            keypoints2d.append([x, y, landmark.visibility])
        return np.array(keypoints2d)


def draw_keypoints(image, keypoints2d, th=0.5):
    if keypoints2d is None:
        return image
    debug_image = image.copy()
    for x,y,confidence in keypoints2d:
        if confidence > th:
            cv.circle(debug_image, (int(x), int(y)), 2, (0, 255, 0), 1)
    return debug_image


if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
    from camera.camera import USB_Camera
    
    camera = USB_Camera('cam 0')
    camera.open(device_id=0)

    pose_estimator = PoseEstimatorMP()

    while True:
        img = camera.get_image()

        results = pose_estimator.process(img)

        #debug_image = draw_keypoints(img, results.keypoints2d)
        debug_image = draw_keypoints(img, results.keypoints2d)
        cv2.imshow('image', debug_image)
         
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()