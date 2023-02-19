import cv2
import mediapipe as mp
from multiprocessing import Process, Manager
import numpy as np
import glob

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

video_path_list = glob.glob('data/*.mp4')

def capture_image(cam_id, return_dict):
    cap = cv2.VideoCapture(cam_id)
    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_height, image_width = image.shape[:2]
            keypoints2d = []
            for landmark in results.pose_landmarks.landmark:
                x = landmark.x * image_width
                y = landmark.y * image_height
                keypoints2d.append([x, y, landmark.visibility])

            return_dict[cam_id] = image, keypoints2d

        


if __name__ == '__main__':
    manager = Manager()
    return_dict = manager.dict()
    processes = []
    
    for video_path in video_path_list[1:]:
        p = Process(target=capture_image, args=(video_path, return_dict))
        processes.append(p)
        p.start()

    while True:
        for video_path in video_path_list[1:]:
            if video_path in return_dict:
                image, keypoints2d = return_dict[video_path]
                # ここで各カメラごとの姿勢推定結果を利用する
                cv2.imshow(str(video_path), image)
        if cv2.waitKey(1) == 27:
            break

    for p in processes:
        p.terminate()