'''
Camera Object class to get image from USB camera and M5 Wi-Fi camera
Camera instance need its name, device ID for USB Camera, Host IP Address for Wi-Fi camera

function:
    __init__(name, args*) : 
    open() : activate camera. return success or not
    close() : deactivate camera. 
    get_image() : get raw image (data type is numpy.uint8). if camera cannot get image, return None
'''


import cv2
import numpy as np
import requests
from enum import Enum
from camera.setting import CameraSetting

# カメラの種類
class CameraType(Enum):
    none = 0
    USB = 1
    M5 = 2
    Video = 3


class M5_Camera:
    def __init__(self, name, host, port=80):
        self.name = name
        self.type = CameraType.M5
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/capture"
        self.isActive = False
        self.camera_setting = CameraSetting()

    def open(self):
        if self.isActive:
            print(f'{self.name} has already opened')
            return True
        self.session = requests.session()
        self.isActive = True
        # set camera setting
        if self.camera_setting.image_height == 0 and self.camera_setting.image_width == 0:
            img = self.get_image()
            if img is not None:
                self.camera_setting.set_intrinsic(image_width=img.shape[1], image_height=img.shape[0])
                print(f'{self.name} is connecting to {self.host}:{self.port}')
            else: # initialize failed
                self.isActive = False
                print(f'{self.name} cannot connect to {self.host}:{self.port}')
        return self.isActive

    def get_image(self):
        if not self.isActive:
            return None
        try:
            response = self.session.get(self.url, timeout=3.0)
            # bufferをndarrayに高速変換
            img_buf= np.frombuffer(response.content, dtype=np.uint8)
            # 画像をデコード
            img = cv2.imdecode(img_buf, cv2.IMREAD_UNCHANGED)
            # 縦長画像に変換
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            # 0埋め
            #h, w = img.shape[:2]
            #p = (h - w) // 2
            #img = cv2.copyMakeBorder(img, 0, 0, p, p, cv2.BORDER_CONSTANT, (255,255,255))
        except:
            img = None
        # リサイズ
        if img is not None and self.camera_setting.image_width > 0 and self.camera_setting.image_height > 0:
            img = cv2.resize(img, dsize=(self.camera_setting.image_width, self.camera_setting.image_height))
        return img

    def close(self):
        self.isActive = False
        return True


class USB_Camera:
    def __init__(self, name, device_id):
        self.name = name
        self.type = CameraType.USB
        self.device_id = device_id
        self.isActive = False
        self.camera_setting = CameraSetting()

    def open(self):
        if self.isActive:
            print(f'{self.name} has already opened')
            return True
        self.cap = cv2.VideoCapture(self.device_id)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  
        self.isActive = True
        # set camera setting
        try:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.camera_setting.set_intrinsic(image_width=width, image_height=height)
        except:
            self.isActive = False
        return self.isActive

    def get_image(self):
        if not self.isActive:
            return None
        try:
            ret, frame = self.cap.read()
        except:
            pass
        img = frame
        if self.camera_setting.image_width > 0 and self.camera_setting.image_height > 0:
            img = cv2.resize(frame, dsize=(self.camera_setting.image_width, self.camera_setting.image_height))
        return img

    def close(self):
        if self.cap is not None:
            self.cap.release()
        self.isActive = False
        return True


class Video:
    def __init__(self, name, video_path):
        self.name = name
        self.type = CameraType.Video
        self.video_path = video_path
        self.isActive = False
        self.camera_setting = CameraSetting()
        self.image = None

    def open(self):
        if self.isActive:
            print(f'{self.name} has already opened')
            return True
        self.cap = cv2.VideoCapture(self.video_path)        
        # set camera setting
        try:
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.camera_setting.set_intrinsic(image_width=width, image_height=height)
            self.isActive = True
        except:
            self.isActive = False
        return self.isActive
        
    def get_image(self):
        if not self.isActive:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        img = frame
        if self.camera_setting.image_width > 0 and self.camera_setting.image_height > 0:
            img = cv2.resize(frame, dsize=(self.camera_setting.image_width, self.camera_setting.image_height))
            return img
        else:
            return frame

    def close(self):
        if self.cap is not None:
            self.cap.release()
        self.isActive = False
        return True

    def set_index(self, idx):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)


if __name__ == '__main__':
    camera = M5_Camera(name='camera', host="192.168.11.3")
    camera.open()

    while True:
        image = camera.get_image()

        if image is not None:
            cv2.imshow(camera.name, image)

        if cv2.waitKey(1) == 27:
            break

    camera.close()