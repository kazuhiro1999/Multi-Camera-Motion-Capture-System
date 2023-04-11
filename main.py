'''
Main program for motion-capture
'''



import cv2
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor

from camera.camera import USB_Camera, Video
from pose.mp_pose import PoseEstimatorMP
from pose.pose3d import recover_pose_3d
from tools.time_utils import TimeUtil

from gui import MainWindow, get_device_config, open_editor
from core import Controller, Pipeline, init_config
from tools.visualization import DebugMonitor, draw_keypoints
from tools.inspect import FPSCalculator


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, default="config.json")
    args = parser.parse_args()

    return args

def main():
    args = get_args()

    controller = Controller()
    controller.load(args.config_path)

    window = MainWindow(controller)
    window.open()
    
    monitor = DebugMonitor()

    keypoints2d_list = []

    # マルチスレッド
    with ThreadPoolExecutor(max_workers=4) as pool:
        while True:
            FPSCalculator.start('main')
            # 3d pose estimation
            proj_matrices = [pipeline.camera.camera_setting.proj_matrix for pipeline in controller.pipelines]
            future_keypoints3d = pool.submit(recover_pose_3d, proj_matrices, keypoints2d_list)
            
            t = TimeUtil.get_time()
            # get images
            future_images = []
            for pipeline in controller.pipelines:
                future = pool.submit(pipeline.camera.get_image)
                future_images.append(future)

            # 2d pose estimation
            images = []
            future_keypoints2d_list = []
            for future_image, pipeline in zip(future_images, controller.pipelines): 
                image = future_image.result()
                images.append(image)
                if pipeline.pose_estimator is not None:
                    future = pool.submit(pipeline.pose_estimator.process, image)
                    future_keypoints2d_list.append(future)

            keypoints2d_list = []
            for future_keypoints in future_keypoints2d_list:
                keypoints2d = future_keypoints.result()
                keypoints2d_list.append(keypoints2d)

            # get 3d pose estimation result
            keypoints3d = future_keypoints3d.result()

            if controller.isActive:
                ret = controller.send(t, keypoints3d)
                if ret:
                    monitor.add_line(t)            

            # debug
            for pipeline, image, keypoints2d in zip(controller.pipelines, images, keypoints2d_list):
                debug_image = draw_keypoints(image, keypoints2d)
                cv2.imshow(pipeline.name, debug_image)

            # fps  
            FPSCalculator.end('main')
            t_exec = FPSCalculator.get_execution_time('main', duration=10)
            print(f"\rFPS:{1/t_exec if t_exec > 0.001 else 0}", end='')

            # window
            event, values = window.window.read(timeout=0)
            if event == '-Add-':
                # get device setting
                camera_config = get_device_config()
                if camera_config is not None:                
                    config = init_config()
                    config['camera'] = camera_config
                    # check resources 
                    if controller.exists(config):
                        print("already exists")
                    else:
                        print("pipeline opened")
                        pipeline = Pipeline(config)
                        controller.add_pipeline(pipeline)
                        window.reload_list()
                        open_editor(pipeline) # initial setting
            if event == '-List-Edit-':
                pipeline = window.get_selected()
                open_editor(pipeline)
            if event == '-List-Delete-':   
                pipeline = window.get_selected()         
                controller.remove_pipeline(pipeline)
                window.reload_list()
                print('deleted')
            if event == '-ModelType-':
                window.set_pose_estimator()
            if event == '-Start-':
                window.start_capture()  
            if event == 'Open Momitor':
                pass          
                
            if event is None:
                cv2.destroyAllWindows()
                break

            monitor.update()

    for pipeline in controller.pipelines:
        pipeline.camera.close()

    controller.save(args.config_path)
    return


if __name__ == '__main__':
    main()