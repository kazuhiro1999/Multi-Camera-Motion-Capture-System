'''
User Interface program for setup motion-capture-system
MainWindow is a main control panel, which set cameras, pose estimation type and udp port
CameraEditor is a editor for camera, which set camera device, segmentation method and camera setting
'''


import PySimpleGUI as sg
import cv2
import numpy as np

from camera.camera import M5_Camera, USB_Camera, CameraType
from segmentation.setting import SegmentationMethod
from segmentation.background_subtraction import BackgroundSubtractor
from segmentation.segmentation import PaddleSegmentation, MediaPipeSelfieSegmentation, DeepLabV3
from pose.setting import ModelType
from controller import Controller

sg.theme("DarkBlue")


class MainWindow:
    def __init__(self, controller: Controller):
        self.controller = controller
        self.isActive = False
        self.window = None

    def open(self):
        model_type = ModelType.none if self.controller.pose_estimator is None else self.controller.pose_estimator.type
        udp_port = self.controller.udp_server.port
        layout = [
            [sg.Menu([['Tool',['Calibrate Cameras (Auto)']]], key='-Menu-')],
            [sg.Text('Cameras')],
            [sg.Listbox(self.controller.get_camera_list(), size=(28,4), key='-CameraList-')],
            [sg.Button('+ Add Camera', size=(26,1), enable_events=True, key='-Add-')],
            # 姿勢推定の設定
            [sg.Frame(title='Configuration', layout=[
                [sg.Text('Model Type', size=(10,1)), sg.Combo([modelType.name for modelType in ModelType], default_value=model_type.name, size=(13,1), readonly=True, enable_events=True, key='-ModelType-')],
                [sg.Text('UDP Port', size=(10,1)), sg.Input(default_text=str(udp_port), size=(15,1), key='-Port-')]])
                ], 
            [sg.Button('Start Capture', size=(26,1), enable_events=True, key='-Start-')]
            ]
        self.window = sg.Window('Motion Capture System', layout=layout, finalize=True)
        self.window['-CameraList-'].bind('<Double-Button>', 'Edit-')
        self.isActive = True
        self.load_config()

    def load_config(self):
        return

    def reload_camera_list(self):
        if not self.isActive:
            return 
        camera_list = self.controller.get_camera_list()
        self.window['-CameraList-'].update(values=camera_list)

    def open_estimator(self):
        if not self.isActive:
            return None
        name = self.window['-ModelType-'].get()
        model_type = ModelType[name]
        self.controller.set_pose_estimator(model_type)
        return self.controller.pose_estimator

    def start_capture(self):
        try:
            port = int(self.window['-Port-'].get())
            self.controller.udp_server.open(port)
            self.controller.isActive = True
            self.window['-Start-'].update(text=f"Started at Port:{port}", disabled=True)
            return True
        except:
            return False

    def get_selected_camera(self):
        name = self.window['-CameraList-'].get()[0]
        camera = self.controller.get_camera(name)
        return camera

    def close(self):
        if self.window is not None:
            self.window.close()



class CameraEditor:

    def __init__(self):
        self.camera = None
        self.isActive = False
        self.window = None

    def open(self, camera=None):
        self.camera = camera
        layout = [
            # デバイス設定
            [sg.Frame(title='Device', layout=[
                [sg.Text('Name', size=(10,1)), sg.Input(default_text='camera', size=(15,1), enable_events=True, key='-Name-')],
                [sg.Text('Camera Type', size=(10,1)), sg.Combo([cameraType.name for cameraType in CameraType], default_value=CameraType.none.name, size=(13,1), readonly=True, enable_events=True, key='-CameraType-')],
                [sg.Text('Device ID', size=(10,1)), sg.Input(default_text='0', size=(15,1), enable_events=True, key='-DeviceID-')],
                [sg.Text('Host IP', size=(10,1)), sg.Input(default_text='192.168.0.0', size=(15,1), enable_events=True, key='-HostIP-')]])
                ], 
            # カメラを開く
            [sg.Button('Open Camera', size=(25,1), enable_events=True, key='-Open-')],
            # セグメンテーション設定
            [sg.Text('Segmentation', size=(10,1)), sg.Combo([method.name for method in SegmentationMethod], default_value=SegmentationMethod.none.name, size=(13,1), readonly=True, enable_events=True, key='-SegmentationMethod-')],
            [sg.Button('Activate Segmentation', size=(25,1), enable_events=True, key='-Activate-')],
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
        self.window = sg.Window('カメラ設定', layout=layout, finalize=True)
        self.isActive = True
        self.load_config(self.camera)

    def load_config(self, camera):
        # device setting
        if camera is None or not self.isActive:
            return
        self.window['-Name-'].update(value=camera.name)
        self.window['-CameraType-'].update(value=camera.type.name)
        if camera.type == CameraType.USB:
            self.window['-DeviceID-'].update(value=camera.device_id)
        elif camera.type == CameraType.M5:
            self.window['-HostIP-'].update(value=camera.host)      
        # segmentation method
        if camera.segmentation is not None:
            self.window['-SegmentationMethod-'].update(value=camera.segmentation.type.name)      
        # update camera_setting
        self.load_camera_setting(self.camera.camera_setting)

    def get_params(self):
        params = {
            'Name': self.window['-Name-'].get(),
            'CameraType': CameraType[self.window['-CameraType-'].get()],
            'DeviceID': int(self.window['-DeviceID-'].get()),
            'Host': self.window['-HostIP-'].get()
        }
        return params

    def open_camera(self):
        if not self.isActive:
            return None
        self.close_camera()
        name = self.window['-Name-'].get()
        cameraType = CameraType[self.window['-CameraType-'].get()]
        if cameraType == CameraType.none:
            self.camera = None
        elif cameraType == CameraType.USB:
            device_id = int(self.window['-DeviceID-'].get())
            usb_camera = USB_Camera(name, device_id)
            usb_camera.open()
            self.camera = usb_camera
        elif cameraType == CameraType.M5:
            host = self.window['-HostIP-'].get()
            m5_camera = M5_Camera(name, host)
            m5_camera.open()
            self.camera = m5_camera
        elif cameraType == CameraType.Video:
            self.camera = None
        else:
            return None
        # update camera_setting
        if self.camera is not None:
            self.load_config(self.camera)
        return self.camera

    def activate_segmentation(self, camera):
        if camera is None or not self.isActive:
            return False
        method = SegmentationMethod[self.window['-SegmentationMethod-'].get()]
        if method == SegmentationMethod.none:
            camera.segmentation = None
        elif method == SegmentationMethod.Subtraction:
            setup_subtractor(camera)
        elif method == SegmentationMethod.Paddle:
            camera.segmentation = PaddleSegmentation()
        elif method == SegmentationMethod.Mediapipe:
            camera.segmentation = MediaPipeSelfieSegmentation()
        elif method == SegmentationMethod.DeepLabV3:
            camera.segmentation = DeepLabV3()
        else:
            return False
        return True

    def apply_camera_setting(self, camera):
        if camera is None or not self.isActive:
            return
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

        camera.camera_setting.set_intrinsic(image_width=image_width, image_height=image_height, FOV=FOV)
        camera.camera_setting.set_transform(position=position, rotation=rotation)
        return True

    def load_camera_setting(self, camera_setting):
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

    def close_camera(self):
        if self.camera is not None:
            self.camera.close()
    
    def close(self):
        if self.window is not None:
            self.window.close()


def open_camera():
    camera = None
    camera_editor = CameraEditor()
    camera_editor.open()
    while True:
        event, values = camera_editor.window.read(timeout=0)
        if event == '-Open-':
            camera = camera_editor.open_camera()
        if event == '-Activate-':
            camera_editor.activate_segmentation(camera)
        if event == '-Apply-':
            camera_editor.apply_camera_setting(camera)
        if event == '-OK-':
            camera.name = camera_editor.window['-Name-'].get() # 名前変更だけ適用
            break                
        if event is None or event == '-Cancel-':
            if camera is not None:
                camera.close()
                cv2.destroyAllWindows()
            camera = None
            break
        if camera is not None:
            img = camera.get_image()
            if img is not None:
                if camera.segmentation is not None:
                    mask = camera.segmentation.process(img)
                    img = (img * mask).astype(np.uint8)
                cv2.imshow(camera.name, img)
    camera_editor.close()
    return camera

def edit_camera(camera):
    _camera = camera
    camera_editor = CameraEditor()
    camera_editor.open(camera)
    while True:
        event, values = camera_editor.window.read(timeout=0)
        if event == '-Open-':
            camera = camera_editor.open_camera()
        if event == '-Activate-':
            camera_editor.activate_segmentation(camera)
        if event == '-Apply-':
            camera_editor.apply_camera_setting(camera)
        if event == '-OK-':            
            camera.name = camera_editor.window['-Name-'].get() # 名前だけ適用
            break                
        if event is None or event == '-Cancel-':
            camera = _camera
            if not camera.isActive:
                camera.open()
            break                
        if camera is not None:
            img = camera.get_image()
            if img is not None:
                if camera.segmentation is not None:
                    mask = camera.segmentation.process(img)
                    img = (img * mask).astype(np.uint8)
                cv2.imshow(camera.name, img)
    camera_editor.close()
    return camera


def setup_subtractor(camera):
    if camera is None:
        return False
    elif camera.segmentation is not None and camera.segmentation.type == SegmentationMethod.Subtraction:
        subtractor = camera.segmentation
    else:
        subtractor = BackgroundSubtractor()
    layout = [
        # 背景設定
        [sg.Button('Shoot', size=(10,1), enable_events=True, key='-Shoot-')],
        # 背景差分パラメータ
        [sg.Frame(title='Subtraction Setting', layout=[
            [sg.Text('Brightness', size=(10,1)), sg.Slider(range=(0.0,1.0), default_value=subtractor.a_min, resolution=0.01, orientation='h', enable_events=True, key='-A-')],
            [sg.Text('Color', size=(10,1)), sg.Slider(range=(0.0,30.0), default_value=subtractor.c_max, resolution=1.0, orientation='h', enable_events=True, key='-C-')],
        ])],
        [sg.Button('Cancel', size=(10,1), enable_events=True, key='-Cancel-'), sg.Button('OK', size=(10,1), pad=((30,0),(0,0)), enable_events=True, key='-OK-')]
    ]
    window = sg.Window(title='背景差分設定', layout=layout, finalize=True)
    while True:
        img = camera.get_image()
        mask = subtractor.process(img)
        debug_image = (img * mask).astype(np.uint8)
        cv2.imshow(camera.name, debug_image)

        event, values = window.read(timeout=0)
        if event == '-Shoot-':
            subtractor.set_background(img)
            cv2.imshow('background', img)
        if event == '-A-':
            subtractor.a_min = values['-A-']
        if event == '-C-':
            subtractor.c_max = values['-C-']
        if event == '-OK-':
            camera.segmentation = subtractor
            break
        if event is None or event == '-Cancel-':
            break
    window.close()
    return True


if __name__ == '__main__':
    #camera = USB_Camera('cam 0')
    #camera.open(device_id=0)
    camera = None

    camera_editor = CameraEditor(camera)
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
