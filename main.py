
import time
import PySimpleGUI as sg
import cv2
import numpy as np

from camera.editor import CameraEditor
from controller import Controller
from pose.pose3d import recover_pose_3d
from tools.visualization import DebugMonitor, draw_keypoints
from pose.setting import ModelType
from segmentation.setting import SegmentationMethod

sg.theme("DarkBlue")


class MainWindow:
    def __init__(self, controller: Controller):
        self.controller = controller
        self.isActive = False
        self.window = None

    def open(self):
        segmentation_method = SegmentationMethod.none if self.controller.segmentation is None else self.controller.segmentation.type
        model_type = ModelType.none if self.controller.pose_estimator is None else self.controller.pose_estimator.type
        udp_port = self.controller.udp_server.port
        layout = [
            [sg.Menu([['Tool',['Calibrate Cameras (Auto)']]], key='-Menu-')],
            [sg.Text('Cameras')],
            [sg.Listbox(self.controller.get_camera_list(), size=(28,4), key='-CameraList-')],
            [sg.Button('+ Add Camera', size=(26,1), enable_events=True, key='-Add-')],
            # 姿勢推定の設定
            [sg.Frame(title='Configuration', layout=[
                [sg.Text('Segmentation', size=(10,1)), sg.Combo([method.name for method in SegmentationMethod], default_value=segmentation_method.name, size=(13,1), readonly=True, enable_events=True, key='-SegmentationMethod-')],
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

    def open_segmentation(self):
        if not self.isActive:
            return None
        name = self.window['-SegmentationMethod-'].get()
        method = SegmentationMethod[name]
        self.controller.set_segmentation(method)
        return self.controller.segmentation

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


def main():
    controller = Controller()
    controller.load_config('config.json')
    window = MainWindow(controller)
    window.open()
    
    camera_editor = CameraEditor()
    monitor = DebugMonitor()

    while True:
        t = time.time()
        proj_matrices = []
        keypoints2d_list = []
        for camera in controller.cameras:
            image = camera.get_image()
            if image is None:
                continue

            input_image = image.copy()
            # person segmentation
            if controller.segmentation is not None:
                mask = controller.segmentation.process(input_image) 
                input_image = (input_image * mask).astype(np.uint8)

            debug_image = input_image
            # 2d pose estimation
            if controller.pose_estimator is not None:
                keypoints = controller.pose_estimator.process(input_image) 
                proj_matrix = camera.camera_setting.get_projection_matrix()

                if keypoints is not None and proj_matrix is not None:
                    keypoints2d_list.append(keypoints)
                    proj_matrices.append(proj_matrix)

                if keypoints is not None:                    
                    debug_image = draw_keypoints(debug_image, keypoints)              

            cv2.imshow(camera.name, debug_image)

        keypoints3d = recover_pose_3d(proj_matrices, keypoints2d_list) # 3d pose estimation
        if controller.isActive:
            controller.send(t, keypoints3d)
            monitor.add_line(t)

        # space calibrator
        #if calibrator.isActive:
        #   calibrator.process(keypoints2d_list)
            
        # window
        event, values = window.window.read(timeout=0)
        if event == '-Add-':
            camera = None
            camera_editor.open(None)
            while True:
                cam_event, values = camera_editor.window.read(timeout=0)
                if cam_event == '-Open-':
                    params = camera_editor.get_params()
                    if not controller.exists(params):
                        camera = camera_editor.open_camera()
                if cam_event == '-Apply-':
                    camera_editor.apply_camera_setting(camera)
                if cam_event == '-OK-':
                    controller.add_camera(camera)
                    window.reload_camera_list()
                    break                
                if cam_event is None or cam_event == '-Cancel-':
                    if camera is not None:
                        camera.close()
                        cv2.destroyAllWindows()
                    break
                if camera_editor.camera is not None:
                    img = camera_editor.camera.get_image()
                    if img is not None:
                        cv2.imshow(camera_editor.camera.name, img)
            camera_editor.close()
        if event == '-CameraList-Edit-':
            selected_camera = window.get_selected_camera()
            camera_editor.open(selected_camera)
            camera = selected_camera
            while True:
                cam_event, values = camera_editor.window.read(timeout=0)
                if cam_event == '-Open-':
                    camera = camera_editor.open_camera()
                if cam_event == '-Apply-':
                    camera_editor.apply_camera_setting(camera)
                if cam_event == '-OK-':
                    if camera is None: # None指定でカメラを削除
                        ret = controller.delete_camera(selected_camera)
                    else: # カメラの設定を変更する
                        ret = controller.replace_camera(selected_camera, camera)
                    if ret:
                        window.reload_camera_list()
                        break                
                if cam_event is None or cam_event == '-Cancel-':
                    if camera is not selected_camera:
                        if camera is not None:
                            camera.close()
                            cv2.destroyWindow(camera.name)
                        selected_camera.open()
                    break
                if camera_editor.camera is not None:
                    img = camera_editor.camera.get_image()
                    cv2.imshow(camera_editor.camera.name, img)
            camera_editor.close()
        if event == '-SegmentationMethod-':
            window.open_segmentation()
        if event == '-ModelType-':
            window.open_estimator()
        if event == '-Start-':
            window.start_capture()
            monitor.open()
            
        if event is None:
            cv2.destroyAllWindows()
            break

        monitor.update()

    controller.save_config('config.json')
    return


if __name__ == '__main__':
    main()