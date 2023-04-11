'''
User Interface program for setup motion-capture-system
MainWindow is a main control panel, which set cameras, pose estimation type and udp port
CameraEditor is a editor for camera, which set camera device, segmentation method and camera setting
'''


import PySimpleGUI as sg
import cv2
import numpy as np

from camera.camera import M5_Camera, USB_Camera, CameraType
from camera.setting import CameraSetting
from segmentation.setting import SegmentationMethod
from segmentation.background_subtraction import BackgroundSubtractor, setup_subtractor
from segmentation.segmentation import PaddleSegmentation, MediaPipeSelfieSegmentation, DeepLabV3
from pose.setting import ModelType
from core import Controller, Pipeline, init_config
from tools.visualization import draw_keypoints

sg.theme("DarkBlue")


class MainWindow:
    def __init__(self, controller: Controller):
        self.controller = controller
        self.isActive = False
        self.window = None

    def open(self):
        model_type = self.controller.model_type
        udp_host = self.controller.udp_client.host
        udp_port = self.controller.udp_client.port
        layout = [
            [sg.Menu([['Tool',['Calibrate Cameras (Auto)']]], key='-Menu-')],
            [sg.Text('Cameras')],
            [sg.Listbox(self.controller.get_name_list(), size=(28,4), key='-List-')],
            [sg.Button('+ Add Camera', size=(26,1), enable_events=True, key='-Add-')],
            # 姿勢推定の設定
            [sg.Frame(title='Configuration', layout=[
                [sg.Text('Model Type', size=(10,1)), sg.Combo([modelType.name for modelType in ModelType], default_value=model_type.name, size=(13,1), readonly=True, enable_events=True, key='-ModelType-')],
                [sg.Text('UDP Host', size=(10,1)), sg.Input(default_text=str(udp_host), size=(15,1), key='-Port-')],
                [sg.Text('UDP Port', size=(10,1)), sg.Input(default_text=str(udp_port), size=(15,1), key='-Port-')]])
                ], 
            [sg.Button('Start Capture', size=(26,1), enable_events=True, key='-Start-')]
            ]
        self.window = sg.Window('Motion Capture System', layout=layout, finalize=True)
        self.window['-List-'].bind('<Double-Button>', 'Edit-')
        self.window['-List-'].bind('<KeyRelease-Delete>', 'Delete-')
        self.isActive = True
        self.load_config()

    def load_config(self):
        return

    def reload_list(self):
        if not self.isActive:
            return 
        name_list = self.controller.get_name_list()
        self.window['-List-'].update(values=name_list)

    def set_pose_estimator(self):
        if not self.isActive:
            return None
        name = self.window['-ModelType-'].get()
        for pipeline in self.controller.pipelines:
            pipeline.set_pose_estimator(name)
        return 

    def start_capture(self):
        try:
            host = self.window['-Host-'].get()
            port = int(self.window['-Port-'].get())
            self.controller.udp_client.open(host=host, port=port)
            self.controller.isActive = True
            self.window['-Start-'].update(text=f"Started at Port:{port}", disabled=True)
            return True
        except:
            return False

    def get_selected(self):
        name = self.window['-List-'].get()[0]
        for pipeline in self.controller.pipelines:
            if pipeline.name == name:
                return pipeline
        return None

    def close(self):
        if self.window is not None:
            self.window.close()


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
    
    def close(self):
        if self.window is not None:
            self.window.close()


# カメラ設定UI
def open_editor(pipeline:Pipeline):
    editor = Editor()
    editor.open(pipeline.camera, pipeline.config) 
    while True:
        event, values = editor.window.read(timeout=0)
        if event == '-SegmentationMethod-':
            method = editor.get_segmentation_method()
            pipeline.set_segmentation(method.name)            
            if method == SegmentationMethod.Subtraction:
                setup_subtractor(pipeline.camera, pipeline.segmentation)
        if event == '-Apply-':
            editor.apply_camera_setting(pipeline.camera.camera_setting)
        if event == '-OK-':
            name = editor.window['-Name-'].get()
            pipeline.camera.name = name
            pipeline.name = name
            pipeline.config['camera']['name'] = name
            ret = True
            break                
        if event is None or event == '-Cancel-':
            ret = False
            break
        # debug image
        if pipeline.camera is not None:
            img = pipeline.camera.get_image()
            if img is None:
                continue
            if pipeline.segmentation is not None:
                mask = pipeline.segmentation.process(img)
                img = (img * mask).astype(np.uint8)
            keypoints = None
            if pipeline.pose_estimator is not None:
                keypoints = pipeline.pose_estimator.process(img) 
            if keypoints is not None:
                img = draw_keypoints(img, keypoints) 
            cv2.imshow(pipeline.name, img)

    editor.close()
    return ret



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
    #camera = USB_Camera('cam 0')
    #camera.open(device_id=0)
    camera = None

    camera_editor = Editor(camera)
    camera_editor.open()
    
    while True:
        event, values = camera_editor.window.read(timeout=0)
        if event is None:
            camera_editor.close()
            break
        if event == '-Open-':
            camera = camera_editor.open_camera()
        if event == '-Apply-':
            camera_editor.apply_camera_setting(camera)
        
        if camera is not None:
            img = camera.get_image()
            cv2.imshow(camera.name, img)
