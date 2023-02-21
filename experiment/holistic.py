import cv2
import numpy as np
import mediapipe as mp
import time

if __name__ == '__main__':
    cap = cv2.VideoCapture("data/0_keito_1_1.mp4")
    t_list = []
    with mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            t0 = time.time()
            results = holistic.process(image)
            t_list.append(time.time() - t0)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_height, image_width = image.shape[:2]
            keypoints2d = []
            for landmark in results.pose_landmarks.landmark:
                x = landmark.x * image_width
                y = landmark.y * image_height
                keypoints2d.append([x, y, landmark.visibility])

            cv2.imshow('image', frame)

            if cv2.waitKey(1) == 27:
                break

            if len(t_list) > 100:
                break

    print(np.array(t_list).mean())