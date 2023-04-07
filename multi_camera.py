import json
import time
import cv2
import os
import argparse
import numpy as np
import PySimpleGUI as sg
from network.udp import UDPClient, UDPServer
from pipeline import Pipeline, get_device_config, init_config
from pose.mp_pose import PoseEstimatorMP
from pose.pose3d import recover_pose_3d
from pose.setting import ModelType
from tools.time_utils import TimeUtil

sg.theme("DarkBlue")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, default="config.json")
    args = parser.parse_args()

    return args


class Controller:

    def __init__(self):
        self.isActive = False
        self.pipelines = []
        self.udp_client = UDPClient()

    # check device has already exist or not
    def exists(self, config):
        for pipeline in self.pipelines:
            cfg = pipeline.cfg
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
        # 名前重複がないか確認
        name = pipeline.cfg['camera']['name']
        if name in self.get_name_list():
            print(f'This Name has already used : {name}')
            return False        
        self.pipelines.append(pipeline)
        return True

    def remove_pipeline(self, pipeline):
        name = pipeline.cfg['camera']['name'] 
        for i, _pipeline in enumerate(self.pipelines):
            if _pipeline.cfg['camera']['name'] == name:
                _pipeline.close()
                self.pipelines.pop(i)
                return True
        return False

    def delete_camera(self, name):
        for i, pipeline in enumerate(self.pipelines):
            if pipeline.cfg['camera']['name'] == name:
                pipeline.close()
                self.pipelines.pop(i)
                return True
        return False

    def get_name_list(self):
        return [pipeline.cfg['camera']['name'] for pipeline in self.pipelines]

    def get_model_type(self):
        if len(self.pipelines) == 0:
            return ModelType.none.name
        model_type = self.pipelines[0].cfg['pose_estimation']
        model_type_list = [pipeline.cfg['pose_estimation'] for pipeline in self.pipelines]
        if model_type_list.count(model_type) == len(model_type_list):
            return model_type
        else:
            return ModelType.none.name

    def get_pipeline(self, name):
        for pipeline in self.pipelines:
            if pipeline.cfg['camera']['name'] == name:
                return pipeline
        return None

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
    
    def save(self, config_path):
        config = {'pipelines':[]}
        for pipeline in self.pipelines:
            cfg = pipeline.get_config()
            config['pipelines'].append(cfg)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        return True

    def load(self, config_path):
        if not os.path.exists(config_path):
            return False
        with open(config_path, 'r') as f:
            config = json.load(f)
        for cfg in config['pipelines']:
            pipeline = Pipeline(cfg)
            self.add_pipeline(pipeline)
        return True


class MainWindow:
    def __init__(self, controller: Controller):
        self.controller = controller
        self.isActive = False
        self.window = None

    def open(self):
        udp_host = self.controller.udp_client.host
        udp_port = self.controller.udp_client.port
        layout = [
            [sg.Menu([['Tool',['Calibrate Cameras (Auto)', 'Open Monitor']]], key='-Menu-')],
            [sg.Text('Cameras')],
            [sg.Listbox(self.controller.get_name_list(), size=(28,4), key='-List-')],
            [sg.Button('+ Add Camera', size=(26,1), enable_events=True, key='-Add-')],
            # 姿勢推定の設定
            [sg.Frame(title='Configuration', layout=[
                [sg.Text('UDP Host', size=(10,1)), sg.Input(default_text=str(udp_host), size=(15,1), key='-Host-')],
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
        pipeline = self.controller.get_pipeline(name)
        return pipeline

    def close(self):
        if self.window is not None:
            self.window.close()



if __name__ == '__main__':
    print("process started : multi_camera.py")
    args = get_args()
    config_path = args.config_path
    controller = Controller()
    controller.load(config_path)

    window = MainWindow(controller)
    window.open()
    
    t_next = TimeUtil.get_time()

    # pipeline setting
    #config = init_config(name='usb camera 1', camera_type='USB', device_id=0)
    #pipeline1 = Pipeline(config)
    #pipelines.append(pipeline1)
    
    while True:
        # update time
        t_now = t_next

        # wait pipeline process
        for pipeline in controller.pipelines:
            pipeline.wait()

        # read data from pipeline
        keypoints2d_list = []
        proj_matrices = []
        for pipeline in controller.pipelines:
            keypoints2d = pipeline.data['keypoints2d']
            if keypoints2d is not None:
                keypoints2d_list.append(keypoints2d)
            proj_matrix = pipeline.data['proj_matrix']
            if proj_matrix is not None:
                proj_matrices.append(proj_matrix)

        # resume pipeline process
        for pipeline in controller.pipelines:
            pipeline.resume()

        # get time
        t_next = TimeUtil.get_time()

        # 3d pose estimation
        keypoints3d = recover_pose_3d(proj_matrices, keypoints2d_list)

        # udp communication        
        if controller.isActive:
            ret = controller.send(t_now, keypoints3d)

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
                    pipeline.edit() # initial setting
        if event == '-List-Edit-':
            pipeline = window.get_selected()
            pipeline.edit()
        if event == '-List-Delete-':   
            pipeline = window.get_selected()         
            controller.remove_pipeline(pipeline)
            window.reload_list()
            print('deleted')
        if event == '-ModelType-':
            print("pass")
        if event == '-Start-':
            for pipeline in controller.pipelines:
                pipeline.reset.set()
            t_start = time.time()
            window.start_capture()  
        if event == 'Open Momitor':
            pass          
            
        if event is None:
            cv2.destroyAllWindows()
            break

        # reload when pipeline config is changed
        for pipeline in controller.pipelines:
            if pipeline.is_changed():
                window.reload_list()

    controller.save(config_path)

    # terminate child processes
    for pipeline in controller.pipelines:
        pipeline.close()

    window.close()

    print("process ended : multi_camera.py")