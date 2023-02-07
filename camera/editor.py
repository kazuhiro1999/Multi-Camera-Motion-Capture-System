import PySimpleGUI as sg
import cv2

from camera.camera import M5_Camera, USB_Camera, CameraType

sg.theme("DarkBlue")


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
            self.load_camera_setting(self.camera.camera_setting)
        return self.camera

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
