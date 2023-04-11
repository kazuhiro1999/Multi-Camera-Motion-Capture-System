import os
import numpy as np
import json
from camera.camera import CameraType, M5_Camera, USB_Camera, Video
from camera.setting import CameraSetting
from pose.mp_holistic import MediapipeHolistic
from pose.mp_pose import PoseEstimatorMP
from pose.ours import PoseNet
from pose.pose3d import recover_pose_3d
from network.udp import UDPClient
from pose.setting import ModelType
from segmentation.background_subtraction import BackgroundSubtractor
from segmentation.segmentation import DeepLabV3, MediaPipeSelfieSegmentation, PaddleSegmentation
from segmentation.setting import SegmentationMethod


def open_camera(config):
    name = config['name']
    camera_type = CameraType[config['type']]
    # camera info
    if camera_type == CameraType.M5:
        host = config['host']
        camera = M5_Camera(name, host=host)
        ret = camera.open()
    elif camera_type == CameraType.USB:
        device_id = config['device_id']
        camera = USB_Camera(name, device_id=device_id)
        ret = camera.open()
    elif camera_type == CameraType.Video:
        video_path = config['video_path'] 
        camera = Video(name, video_path=video_path)
        ret = camera.open()
    else:
        ret = False
        camera = None
    return ret, camera

def open_segmentation(method):
    method = SegmentationMethod[method]
    if method == SegmentationMethod.none:
        return None
    elif method == SegmentationMethod.Mediapipe:
        return MediaPipeSelfieSegmentation()
    elif method == SegmentationMethod.Paddle:
        return PaddleSegmentation()
    elif method == SegmentationMethod.DeepLabV3:
        return DeepLabV3()
    elif method == SegmentationMethod.Subtraction:
        return BackgroundSubtractor()
    return None
    
def open_pose_estimator(model_type):
    model_type = ModelType[model_type]
    if model_type == ModelType.none:
        return None
    elif model_type == ModelType.Humanoid:
        return PoseNet()
    elif model_type == ModelType.MediapipePose:
        return PoseEstimatorMP()
    elif model_type == ModelType.MediapipeHolistic:
        return MediapipeHolistic()
    return None

def get_camera_setting(config):
    FOV = config['FOV']
    image_width = config['image_width']
    image_height = config['image_height']
    position_x = config['position']['x']    
    position_y = config['position']['y']
    position_z = config['position']['z']
    position = np.array([position_x, position_y, position_z])
    rotation_x = config['rotation']['x']  
    rotation_y = config['rotation']['y']  
    rotation_z = config['rotation']['z']  
    rotation = np.array([rotation_x, rotation_y, rotation_z])
    camera_setting = CameraSetting()
    camera_setting.set_intrinsic(FOV=FOV, image_height=image_height, image_width=image_width)
    camera_setting.set_transform(position=position, vector=rotation)
    return camera_setting

def init_config(name="", camera_type='none', device_id=0, host='', segmentation='none', pose_estimation='none', debug=True):
    config = {
        'camera':{
            'name':name,
            'type':camera_type,
            'device_id':device_id, # for usb camera
            'host':host, # for m5 camera
        },
        'camera_setting':{
            "FOV": 90,
            "image_width": 640,
            "image_height": 360,
            "position": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0
            },
            "rotation": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0
            }
        },
        'segmentation':segmentation,
        'pose_estimation':pose_estimation,
        'debug':debug,
    }
    return config


class Pipeline:
    def __init__(self, config):
        ret, camera = open_camera(config['camera'])
        self.camera = camera
        self.name = config['camera']['name']
        self.config = config
        self.segmentation = None
        self.pose_estimator = None
    
    def load_config(self, config):
        self.set_camera_setting(config['camera_setting'])        
        self.set_segmentation(config['segmentation'])
        self.set_pose_estimator(config['pose_estimation'])

    def set_segmentation(self, method):
        self.config['segmentation'] = method
        self.segmentation = open_segmentation(method)

    def set_pose_estimator(self, model_type):
        self.config['pose_estimation'] = model_type
        self.pose_estimator = open_pose_estimator(model_type)

    def set_camera_setting(self, camera_setting_config):
        self.config['camera_setting'] = camera_setting_config
        self.camera.camera_setting = get_camera_setting(camera_setting_config)   

    def get_config(self):
        return self.config
    

class Controller:

    def __init__(self):
        self.isActive = False
        self.pipelines = []
        self.model_type = ModelType.none
        self.udp_client = UDPClient()
    
    def load(self, config_path):
        if not os.path.exists(config_path):
            return False
        with open(config_path, 'r') as f:
            config = json.load(f)
        for cfg in config['pipelines']:
            pipeline = Pipeline(cfg)
            self.add_pipeline(pipeline)
        return True
    
    def save(self, config_path):
        config = {'pipelines':[]}
        for pipeline in self.pipelines:
            cfg = pipeline.config
            config['pipelines'].append(cfg)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    
    # check device has already exist or not
    def exists(self, config):
        for pipeline in self.pipelines:
            cfg = pipeline.config
            if config['camera']['type'] != cfg['camera']['type']:
                continue
            if config['camera']['type'] == 'USB' and config['camera']['device_id'] == cfg['camera']['device_id']:          
                print(f"USB Device {config['camera']['device_id']} has already used")          
                return True
            elif config['camera']['type'] == 'M5' and config['camera']['host'] == cfg['camera']['host']:
                print(f"Host {config['camera']['host']} has already used")
                return True
            else:
                pass
        return False

    def add_pipeline(self, pipeline):       
        self.pipelines.append(pipeline)
        return True

    def remove_pipeline(self, pipeline):
        for i, _pipeline in enumerate(self.pipelines):
            if pipeline.name == _pipeline.name:
                self.pipelines.pop(i)
                return True
        return False

    def get_name_list(self):
        return [pipeline.name for pipeline in self.pipelines]

    def send(self, timestamp, keypoints3d):
        if keypoints3d is None:
            return False
        data = {"Type":self.get_model_type(), "TimeStamp": timestamp, "Bones":[]}
        keys = PoseEstimatorMP.KEYPOINT_DICT
        for key in keys:
            bone = {
                "Name": key,
                "Position":{
                    "x": float(keypoints3d[keys[key],0]),
                    "y": float(keypoints3d[keys[key],1]),
                    "z": float(keypoints3d[keys[key],2]),
                }
            }
            data['Bones'].append(bone)                
        ret = self.udp_client.send(data)
        return ret

if __name__ == '__main__':
    controller = Controller()