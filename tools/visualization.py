import cv2
import numpy as np

def draw_keypoints(image, keypoints2d, th=0.2):
    if keypoints2d is None:
        return image
    debug_image = image.copy()
    for x, y, confidence in keypoints2d:
        if confidence > th:
            cv2.circle(debug_image, (int(x), int(y)), radius=1, color=(0,255,0), thickness=1)
    return debug_image


class DebugMonitor:

    def __init__(self):
        pass

    def open(self, udp_port):
        pass