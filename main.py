'''
Main program for motion-capture
'''


import time
import cv2
import numpy as np

from core import MainWindow, edit_camera, open_camera
from controller import Controller
from pose.pose3d import recover_pose_3d
from tools.visualization import DebugMonitor, draw_keypoints


CONFIG_PATH = 'config.json'


def main():
    controller = Controller()
    controller.load_config(CONFIG_PATH)
    window = MainWindow(controller)
    window.open()
    
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
            if camera.segmentation is not None:
                mask = camera.segmentation.process(input_image) 
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
            camera = open_camera()
            if not controller.exists(camera):
                controller.add_camera(camera)
                window.reload_camera_list()
        if event == '-CameraList-Edit-':
            _camera = window.get_selected_camera()
            camera = edit_camera(_camera)
            if camera is None: # None指定でカメラを削除
                ret = controller.delete_camera(camera)
            else: # カメラの設定を変更する
                ret = controller.replace_camera(_camera, camera)
            if ret:
                window.reload_camera_list()
        if event == '-ModelType-':
            window.open_estimator()
        if event == '-Start-':
            window.start_capture()
            monitor.open()
            
        if event is None:
            cv2.destroyAllWindows()
            break

        monitor.update()

    controller.save_config(CONFIG_PATH)
    return


if __name__ == '__main__':
    main()