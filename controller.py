import numpy as np
import json
from camera.camera import CameraType, M5_Camera, USB_Camera
from pose.mp_pose import PoseEstimatorMP
from pose.ours import PoseNet
from pose.pose3d import recover_pose_3d
from network.udp import UDPServer
from pose.setting import ModelType
from segmentation.background_subtraction import BackgroundSubtractor
from segmentation.segmentation import DeepLabV3, MediaPipeSelfieSegmentation, PaddleSegmentation
from segmentation.setting import SegmentationMethod


class Controller:

    def __init__(self):
        self.isActive = False
        self.cameras = []
        self.segmentation = None
        self.pose_estimator = None
        self.udp_server = UDPServer()
        self.keys = None

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        # for UDP Translation
        port = config['udp_port']
        self.udp_server.open(port=port)
        # for segmentation
        segmentation_method = SegmentationMethod[config['segmentation method']]
        self.set_segmentation(segmentation_method)
        # for pose estimation
        model_type = ModelType[config['model type']]
        self.set_pose_estimator(model_type)
        # for each camera
        for camera_config in config['cameras']:
            name = camera_config['name']
            cameraType = CameraType[camera_config['type']]
            # camera info
            if cameraType == CameraType.M5:
                host = camera_config['host']
                port = camera_config['port']
                camera = M5_Camera(name, host=host, port=port)
                ret = camera.open()
                if ret:
                    self.cameras.append(camera)
            elif cameraType == CameraType.USB:
                device_id = camera_config['device_id']
                camera = USB_Camera(name, device_id=device_id)
                ret = camera.open()
                if ret:
                    self.cameras.append(camera)
            else:
                camera = None
                print(f"cannot load setting : {name}, type={camera_config['type']}")

            # camera settings
            if camera is not None:
                FOV = camera_config['camera_setting']['FOV']
                image_width = camera_config['camera_setting']['image_width']
                image_height = camera_config['camera_setting']['image_height']
                camera.camera_setting.set_intrinsic(image_width=image_width, image_height=image_height, FOV=FOV)
                position_x = camera_config['camera_setting']['position']['x']
                position_y = camera_config['camera_setting']['position']['y']
                position_z = camera_config['camera_setting']['position']['z']
                position = np.array([position_x, position_y, position_z])
                rotation_x = camera_config['camera_setting']['rotation']['x']
                rotation_y = camera_config['camera_setting']['rotation']['y']
                rotation_z = camera_config['camera_setting']['rotation']['z']
                rotation = np.array([rotation_x, rotation_y, rotation_z])
                camera.camera_setting.set_transform(position=position, rotation=rotation)

    def save_config(self, config_path):
        config = {}
        # for UDP Translation
        config['udp_port'] = self.udp_server.port
        # for segmentation
        config['segmentation method'] = self.segmentation.type.name if self.segmentation else SegmentationMethod.none.name
        # for pose estimation
        config['model type'] = self.pose_estimator.type.name if self.pose_estimator else ModelType.none.name
        # for each camera
        config['cameras'] = []
        for camera in self.cameras:
            camera_config = {}
            camera_config['name'] = camera.name
            camera_config['type'] = camera.type.name
            # camera info
            if camera.type == CameraType.M5:
                camera_config['host'] = camera.host
                camera_config['port'] = camera.port
            elif camera.type == CameraType.USB:
                camera_config['device_id'] = camera.device_id
            else:
                pass
            # camera settings
            camera_config['camera_setting'] = {
                'FOV' : camera.camera_setting.fov,
                'image_width' : camera.camera_setting.image_width,
                'image_height' : camera.camera_setting.image_height,
                'position' : {
                    'x' : float(camera.camera_setting.position[0]),
                    'y' : float(camera.camera_setting.position[1]),
                    'z' : float(camera.camera_setting.position[2])}, 
                'rotation' : {
                    'x' : float(camera.camera_setting.rotation[0]),
                    'y' : float(camera.camera_setting.rotation[1]),
                    'z' : float(camera.camera_setting.rotation[2])}
            }
            config['cameras'].append(camera_config)

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

    # judge weather camera has already exist
    def exists(self, params):
        for _camera in self.cameras:
            if params['CameraType'] == _camera.type and params['CameraType'] == CameraType.USB:
                if _camera.device_id == params['DeviceID']:          
                    print(f"USB Device {params['DeviceID']} has already used")          
                    return True
            elif params['CameraType'] == _camera.type and params['CameraType'] == CameraType.M5:
                if _camera.host == params['Host']:
                    print(f"Host {params['Host']} has already used")
                    return True
            else:
                pass
        return False

    def add_camera(self, camera):
        if camera is None:
            return False
        # 名前重複がないか確認
        if camera.name in self.get_camera_list():
            print(f'This Name has already used : {camera.name}')
            return False        
        self.cameras.append(camera)
        return True

    def delete_camera(self, camera):
        for i, _camera in enumerate(self.cameras):
            if camera.name == _camera.name:
                self.cameras.pop(i)
                return True
        return False

    def get_camera_list(self):
        return [camera.name for camera in self.cameras]

    def get_camera(self, name):
        for camera in self.cameras:
            if camera.name == name:
                return camera
        return None

    def replace_camera(self, camera, new_camera):
        for i, _camera in enumerate(self.cameras):
            if _camera.name == camera.name:
                self.cameras[i] = new_camera
                return True
        return False

    def set_segmentation(self, method):        
        if method == SegmentationMethod.Subtraction:
            self.segmentation = BackgroundSubtractor()
            # 背景の設定タイミング
        elif method == SegmentationMethod.Mediapipe:
            self.segmentation = MediaPipeSelfieSegmentation()
        elif method == SegmentationMethod.Paddle:
            self.segmentation = PaddleSegmentation()
        elif method == SegmentationMethod.DeepLabV3:
            self.segmentation = DeepLabV3()
        else:
            self.segmentation = None
        return True

    def set_pose_estimator(self, model_type):        
        if model_type == ModelType.Mediapipe:
            self.pose_estimator = PoseEstimatorMP()
            self.keys = PoseEstimatorMP.KEYPOINT_DICT
        elif model_type == ModelType.Ours:
            self.pose_estimator = PoseNet()
            self.keys = PoseNet.KEYPOINT_DICT
        else:
            self.pose_estimator = None
            self.keys = None
        return True

    def send(self, timestamp, keypoints3d):
        if keypoints3d is None or self.keys is None:
            return False
        data = {'timestamp': timestamp}
        for key in self.keys:
            data[key] = {
                'x': keypoints3d[self.keys[key],0],
                'y': keypoints3d[self.keys[key],1],
                'z': keypoints3d[self.keys[key],2]
            }
        self.udp_server.send(data)
        return True

