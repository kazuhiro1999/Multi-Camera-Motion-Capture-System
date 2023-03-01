import time
import cv2
import mediapipe as mp
from multiprocessing import Process, Manager, Event
import numpy as np
import glob
import PySimpleGUI as sg

from camera.setting import CameraSetting
sg.theme('DarkBlue')

from camera.camera import CameraType, M5_Camera, USB_Camera, Video
from pose.mp_pose import PoseEstimatorMP
from pose.ours import PoseNet
from pose.setting import ModelType
from segmentation.background_subtraction import BackgroundSubtractor, setup_subtractor
from segmentation.segmentation import DeepLabV3, MediaPipeSelfieSegmentation, PaddleSegmentation
from segmentation.setting import SegmentationMethod
from tools.visualization import draw_keypoints
from core import CameraEditor, edit_camera

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

video_path_list = glob.glob('data/*.mp4')


def open_camera(config):
    name = config['name']
    camera_type = CameraType[config['type']]
    # camera info
    if camera_type == CameraType.M5:
        host = config['host']
        port = config['port']
        camera = M5_Camera(name, host=host, port=port)
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
        camera = None
    return camera

def open_segmentation(method):
    method = SegmentationMethod[method]
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

def open_pose_estimator(model_type):
    model_type = ModelType[model_type]
    if model_type == ModelType.Mediapipe:
        pose_estimator = PoseEstimatorMP()
    elif model_type == ModelType.Ours:
        pose_estimator = PoseNet()
    else:
        pose_estimator = None
    return pose_estimator

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

def update_config(config, camera=None, segmentation=None, pose_estimator=None):
    # camera device
    if camera is not None:
        if camera.type == CameraType.USB:
            camera_config = {
                'name':camera.name,
                'type':camera.type.name,
                'device_id':camera.device_id,
            }
        elif camera.type == CameraType.M5:
            camera_config = {
                'name':camera.name,
                'type':camera.type.name,
                'host':camera.host,
            }
        elif camera.type == CameraType.Video:
            camera_config = {
                'name':camera.name,
                'type':camera.type.name,
                'video_path':camera.video_path,
            }
        else:
            camera_config = {
                'name':"undefined",
                'type':CameraType.none.name,
            }
        config['camera'] = camera_config

        # camera setting
        if camera is not None and camera.camera_setting is not None:
            config['camera_setting'] = {
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
        else:
            config['camera_setting'] = {}

        if segmentation is not None:
            config['segmentation'] = segmentation.type.name
        else:
            config['segmentation'] = SegmentationMethod.none.name

        if pose_estimator is not None:
            config['pose_estimation'] = pose_estimator.type.name
        else:
            config['pose_estimation'] = ModelType.none.name



# カメラ設定からedit,プロセススタートまで1つのカメラに関する処理を行う
class Pipeline:
    def __init__(self, config=None):
        # プロセス間通信の設定
        manager = Manager()        
        self.cfg = manager.dict(config)
        self.status = manager.dict({'isActive':True, 'isPlaying':True})
        self.data = manager.dict({'image':None, 'keypoints2d':None, 'proj_matrix':None})
        self.flag = Event() # 同期通信
        self.event = Event() # カメラ設定
        self.changed = Event() # 変更
        self.reset = Event() # 動画を初めから再生
        self.end = Event()

        # 初期設定を開く（+ Add Camera）
        self.process = Process(target=self.start, args=(self.cfg, self.status, self.data, self.flag, self.event, self.changed, self.reset, self.end))
        self.process.start()

    def close(self):
        self.status['isActive'] = False
        self.end.wait()
        self.process.terminate()

    # this method loop on background
    def start(self, config, status, data, flag, event, changed, reset, end):
        # open camera device
        camera = open_camera(config['camera'])
        # open other settings
        segmentation = open_segmentation(config['segmentation'])
        pose_estimator = open_pose_estimator(config['pose_estimation'])

        image = camera.get_image()
        keypoints = None

        while status['isActive']:
            # wait until sync flag is set
            if flag.is_set():
                continue
            if status['isPlaying']:
                # get new image from camera
                image = camera.get_image()
            if image is not None:                
                data['image'] = image
            else:
                flag.set()
                cv2.waitKey(1)
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

            # debug
            if config['debug']:
                debug_image = input_image.copy()
                if keypoints is not None:
                    debug_image = draw_keypoints(debug_image, keypoints) 
                cv2.imshow(camera.name, debug_image)

            # handle editor event
            if event.is_set():
                ret, _segmentation, _pose_estimator = open_editor(camera, config, segmentation, pose_estimator)
                if ret:
                    segmentation = _segmentation
                    pose_estimator = _pose_estimator
                    changed.set()
                event.clear()

            if reset.is_set():
                if camera.type == CameraType.Video:
                    camera.set_index(0)
                reset.clear()

            cv2.waitKey(1)

        camera.close()
        end.set()

    def wait(self):
        self.flag.wait()

    def resume(self):
        self.flag.clear()

    def edit(self):
        self.event.set()

    def is_changed(self):
        changed = self.changed.is_set()
        if changed:
            self.changed.clear()
        return changed
    
    def play(self):
        self.status['isPlaying'] = True
   

class Editor:

    def __init__(self):
        self.isActive = False
        self.window = None

    def open(self, camera=None, config=None):
        layout = [
            # デバイス設定
            [sg.Frame(title='Device', layout=[
                [sg.Text('Name', size=(10,1)), sg.Input(default_text="", size=(15,1), enable_events=True, key='-Name-')],
                [sg.Text('Camera Type', size=(10,1)), sg.Text(CameraType.none.name, size=(15,1), key='-CameraType-')],
                [sg.Text('Device ID', size=(10,1)), sg.Text("", size=(15,1), key='-DeviceID-')],
                [sg.Text('Host IP', size=(10,1)), sg.Text("", size=(15,1), key='-HostIP-')]])
                ], 
            # セグメンテーション設定
            [sg.Text('Segmentation', size=(10,1)), sg.Combo([method.name for method in SegmentationMethod], default_value=SegmentationMethod.none.name, size=(13,1), readonly=True, enable_events=True, key='-SegmentationMethod-')],
            # 姿勢推定設定
            [sg.Text('Pose', size=(10,1)), sg.Combo([model_type.name for model_type in ModelType], default_value=ModelType.none.name, size=(13,1), readonly=True, enable_events=True, key='-PoseEstimation-')],
            # カメラ設定
            [sg.Frame(title='Camera Setting', layout=[
                [sg.Text('Image Size', size=(10,1)), 
                    sg.Input(default_text='0', size=(5,1), enable_events=True, key='-Width-'),
                    sg.Text('×', size=(1,1)), 
                    sg.Input(default_text='0', size=(5,1), enable_events=True, key='-Height-')],
                [sg.Text('FOV', size=(10,1)), sg.Input(default_text='90', size=(15,1), enable_events=True, key='-FOV-')],
                [sg.Text('Position', size=(10,1))], 
                [sg.Text('', size=(1,1)),
                    sg.Text('x', size=(1,1)), sg.Input(default_text='0', size=(4,1), enable_events=True, key='-PositionX-'),
                    sg.Text('y', size=(1,1)), sg.Input(default_text='0', size=(4,1), enable_events=True, key='-PositionY-'),
                    sg.Text('z', size=(1,1)), sg.Input(default_text='0', size=(4,1), enable_events=True, key='-PositionZ-')],
                [sg.Text('Rotation', size=(10,1))], 
                [sg.Text('', size=(1,1)),
                    sg.Text('x', size=(1,1)), sg.Input(default_text='0', size=(4,1), enable_events=True, key='-RotationX-'),
                    sg.Text('y', size=(1,1)), sg.Input(default_text='0', size=(4,1), enable_events=True, key='-RotationY-'),
                    sg.Text('z', size=(1,1)), sg.Input(default_text='0', size=(4,1), enable_events=True, key='-RotationZ-')]])
                ], 
            [sg.Button('Apply Setting', size=(25,1), enable_events=True, key='-Apply-')],
            [sg.Button('Cancel', size=(10,1), enable_events=True, key='-Cancel-'), sg.Button('OK', size=(10,1), pad=((30,0),(0,0)), enable_events=True, key='-OK-')]
            ]
        self.window = sg.Window('Camera Setting', layout=layout, finalize=True)
        self.isActive = True
        self.load_config(config)
        self.load_camera_setting(camera.camera_setting)

    def load_config(self, config):
        # device setting
        if config is None or not self.isActive:
            return
        self.window['-Name-'].update(value=config['camera']['name'])
        self.window['-CameraType-'].update(value=config['camera']['type'])
        camera_type = CameraType[config['camera']['type']]
        if camera_type == CameraType.USB:
            self.window['-DeviceID-'].update(value=config['camera']['device_id'])
        elif camera_type == CameraType.M5:
            self.window['-HostIP-'].update(value=config['camera']['host'])      
        # segmentation method
        self.window['-SegmentationMethod-'].update(value=config['segmentation'])   
        self.window['-PoseEstimation-'].update(value=config['pose_estimation'])      

    def load_camera_setting(self, camera_setting:CameraSetting):
        if not self.isActive:
            return
        self.window['-Width-'].update(value=camera_setting.image_width)
        self.window['-Height-'].update(value=camera_setting.image_height)
        self.window['-FOV-'].update(value=camera_setting.fov)
        if camera_setting.position is not None:
            self.window['-PositionX-'].update(value=camera_setting.position[0])
            self.window['-PositionY-'].update(value=camera_setting.position[1])
            self.window['-PositionZ-'].update(value=camera_setting.position[2])
        if camera_setting.rotation is not None:
            self.window['-RotationX-'].update(value=camera_setting.position[0])
            self.window['-RotationY-'].update(value=camera_setting.rotation[1])
            self.window['-RotationZ-'].update(value=camera_setting.rotation[2])
            
    def apply_camera_setting(self, camera_setting:CameraSetting):
        if camera_setting is None or not self.isActive:
            return False
        image_width = int(self.window['-Width-'].get())
        image_height = int(self.window['-Height-'].get())
        FOV = float(self.window['-FOV-'].get())
        pos_x = float(self.window['-PositionX-'].get())
        pos_y = float(self.window['-PositionY-'].get())
        pos_z = float(self.window['-PositionZ-'].get())
        position = [pos_x, pos_y, pos_z]
        rot_x = float(self.window['-RotationX-'].get())
        rot_y = float(self.window['-RotationY-'].get())
        rot_z = float(self.window['-RotationZ-'].get())
        rotation = [rot_x, rot_y, rot_z]

        camera_setting.set_intrinsic(image_width=image_width, image_height=image_height, FOV=FOV)
        camera_setting.set_transform(position=position, rotation=rotation)
        return True
    
    def get_segmentation_method(self):
        if not self.isActive:
            return None
        method = SegmentationMethod[self.window['-SegmentationMethod-'].get()]
        return method
    
    def get_model_type(self):
        if not self.isActive:
            return None
        model_type = ModelType[self.window['-PoseEstimation-'].get()]
        return model_type
    
    def close(self):
        if self.window is not None:
            self.window.close()

# カメラ設定UI
def open_editor(camera, config=None, segmentation=None, pose_estimator=None):
    editor = Editor()
    editor.open(camera, config) 
    while True:
        event, values = editor.window.read(timeout=0)
        if event == '-SegmentationMethod-':
            method = editor.get_segmentation_method()
            segmentation = open_segmentation(method.name)            
            if method == SegmentationMethod.Subtraction:
                setup_subtractor(camera, segmentation)
        if event == '-PoseEstimation-':
            model_type = editor.get_model_type()
            pose_estimator = open_pose_estimator(model_type.name)
        if event == '-Apply-':
            editor.apply_camera_setting(camera.camera_setting)
        if event == '-OK-':
            camera.name = editor.window['-Name-'].get()
            if config is None:
                config = init_config()
            update_config(config, camera, segmentation, pose_estimator)
            ret = True
            break                
        if event is None or event == '-Cancel-':
            ret = False
            break
        # debug image
        if camera is not None:
            img = camera.get_image()
            if img is None:
                continue
            if segmentation is not None:
                mask = segmentation.process(img)
                img = (img * mask).astype(np.uint8)
            keypoints = None
            if pose_estimator is not None:
                keypoints = pose_estimator.process(img) 
            if keypoints is not None:
                img = draw_keypoints(img, keypoints) 
            cv2.imshow(camera.name, img)

    editor.close()
    return ret, segmentation, pose_estimator


def get_device_config():
    layout = [
        [sg.Text('Name', size=(10,1)), sg.Input(default_text="", size=(15,1), enable_events=True, key='-Name-')],
        [sg.Text('Camera Type', size=(10,1)), sg.Combo([cameraType.name for cameraType in CameraType], default_value=CameraType.none.name, size=(13,1), readonly=True, enable_events=True, key='-CameraType-')],
        [sg.Button('Cancel', size=(10,1), enable_events=True, key='-Cancel-'), sg.Button('OK', size=(10,1), pad=((30,0),(0,0)), enable_events=True, key='-OK-')]
        ]
    window = sg.Window(title="Device Setting", layout=layout, finalize=True)
    while True:
        event, values = window.read(timeout=0)
        if event is None or event == '-Cancel-':
            window.close()
            return None
        if event == '-OK-':
            name = values['-Name-']
            camera_type = CameraType[values['-CameraType-']]
            break
    window.close()
    camera_config = {
        'name':name,
        'type':camera_type.name
    }

    if camera_type == CameraType.USB:
        while True:
            ret = sg.popup_get_text('Device ID', title="Device Setting")
            if ret is None:
                return None
            try:
                device_id = int(ret)
                camera_config['device_id'] = device_id
                break
            except:
                print("device id must be integer")
    elif camera_type == CameraType.M5:
        host = sg.popup_get_text('Host IP', title="Device Setting")
        if host is None:
            return None
        camera_config['host'] = host
    elif camera_type == CameraType.Video:
        video_path = sg.popup_get_file('Video Path', title="Device Setting")
        if video_path is None:
            return None
        else:
            camera_config['video_path'] = video_path
    else:
        return None

    return camera_config        


if __name__ == '__main__':
    pipelines = []

    config = init_config()
    config['camera'] = get_device_config()

    pipeline1 = Pipeline(config)
    pipelines.append(pipeline1)

    
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
    
