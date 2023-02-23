import time
import cv2
import mediapipe as mp
from multiprocessing import Process, Manager, Event
import numpy as np
import glob

from camera.camera import CameraType, M5_Camera, USB_Camera
from pose.mp_pose import PoseEstimatorMP
from pose.ours import PoseNet
from pose.setting import ModelType
from segmentation.background_subtraction import BackgroundSubtractor
from segmentation.segmentation import DeepLabV3, MediaPipeSelfieSegmentation, PaddleSegmentation
from segmentation.setting import SegmentationMethod
from tools.visualization import draw_keypoints

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

video_path_list = glob.glob('data/*.mp4')



def open_camera(config):
    name = config['name']
    cameraType = CameraType[config['type']]
    # camera info
    if cameraType == CameraType.M5:
        host = config['host']
        port = config['port']
        camera = M5_Camera(name, host=host, port=port)
        ret = camera.open()
    elif cameraType == CameraType.USB:
        device_id = config['device_id']
        camera = USB_Camera(name, device_id=device_id)
        ret = camera.open()
    else:
        camera = None
    return camera

def open_segmentation(config):
    method = SegmentationMethod[config['method']]
    if method == SegmentationMethod.none:
        segmentation = None
    elif method == SegmentationMethod.Subtraction:
        segmentation = BackgroundSubtractor()
    elif method == SegmentationMethod.Paddle:
        segmentation = PaddleSegmentation()
    elif method == SegmentationMethod.Mediapipe:
        segmentation = MediaPipeSelfieSegmentation()
    elif method == SegmentationMethod.DeepLabV3:
        segmentation = DeepLabV3()
    else:
        segmentation = None
    return segmentation

def open_pose_estimator(config):
    model_type = ModelType[config['type']]
    if model_type == ModelType.Mediapipe:
        pose_estimator = PoseEstimatorMP()
    elif model_type == ModelType.Ours:
        pose_estimator = PoseNet()
    else:
        pose_estimator = None
    return pose_estimator


class Pipeline:
    def __init__(self):
        self.isActive = Event()
        self.cfg = None
        self.flag = Event() # for sync
        self.status = Manager().dict({'isActive':False})
        self.data = Manager().dict({'image':None, 'keypoints2d':None, 'proj_matrix':None})
        self.process = None

    def open(self, config):
        if self.process is not None:
            return
        # create new process
        self.cfg = config
        self.status['isActive'] = True
        self.process = Process(target=self.start, args=(config, self.status, self.flag, self.data))
        self.process.start()

    def close(self):
        if self.process is None:
            return
        self.status['isActive'] = False
        self.process.terminate()

    # this method loop on background
    def start(self, config, status, flag, data):
        camera = open_camera(config['camera'])
        segmentation = open_segmentation(config['segmentation'])
        pose_estimator = open_pose_estimator(config['pose'])

        while status['isActive']:
            if flag.is_set():
                continue
            # get image from camera
            image = camera.get_image()
            if image is not None:                
                data['image'] = image
            else:
                continue

            # apply segmentation
            input_image = image.copy()
            if segmentation is not None:
                mask = segmentation.process(input_image)                
                input_image = (input_image * mask).astype(np.uint8)

            # estimate pose
            if pose_estimator is not None:
                keypoints = pose_estimator.process(input_image) 
                proj_matrix = camera.camera_setting.get_projection_matrix()

                data['keypoints2d'] = keypoints
                data['proj_matrix'] = proj_matrix                   

            flag.set()

            if config['debug_image']:
                debug_image = input_image.copy()
                if keypoints is not None:
                    debug_image = draw_keypoints(debug_image, keypoints) 
                cv2.imshow(camera.name, debug_image)

            cv2.waitKey(1)

        camera.close()

    def wait(self):
        self.flag.wait()

    def resume(self):
        self.flag.clear()


if __name__ == '__main__':
    pipelines = []

    pipeline1 = Pipeline()
    config1 = {
        'camera':{
            'name': 'camera 1',
            'type': 'USB',
            'device_id': video_path_list[0],
        },
        'segmentation':{
            'method': 'none',
        },
        'pose':{
            'type': 'Mediapipe',
        },
        'debug_image': True
    }
    pipeline1.open(config1)
    pipelines.append(pipeline1)
    
    pipeline2 = Pipeline()
    config2 = {
        'camera':{
            'name': 'camera 2',
            'type': 'USB',
            'device_id': video_path_list[1],
        },
        'segmentation':{
            'method': 'none',
        },
        'pose':{
            'type': 'Mediapipe',
        },
        'debug_image': True
    }
    pipeline2.open(config2)
    pipelines.append(pipeline2)

    while True:
        # wait pipeline process
        for pipeline in pipelines:
            pipeline.wait()

        # read data from pipeline
        keypoints2d_list = []
        proj_matrices = []
        for pipeline in pipelines:
            keypoints2d = pipeline.data['keypoints2d']
            if keypoints2d is not None:
                keypoints2d_list.append(keypoints2d)
            proj_matrix = pipeline.data['proj_matrix']
            if proj_matrix is not None:
                proj_matrices.append(proj_matrix)

        # resume pipeline process
        for pipeline in pipelines:
            pipeline.resume()

        # 3d pose estimation
        keypoints2d_list = np.array(keypoints2d_list)
        print(keypoints2d_list.shape)

        if cv2.waitKey(1) == 27:
            break

    for pipeline in pipelines:
        pipeline.close()