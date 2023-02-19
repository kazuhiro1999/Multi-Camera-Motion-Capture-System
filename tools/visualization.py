import asyncio
import json
import time
import cv2
import numpy as np
import PySimpleGUI as sg

from network.udp import UDPClient

def draw_keypoints(image, keypoints2d, th=0.2):
    if keypoints2d is None:
        return image
    debug_image = image.copy()
    for x, y, confidence in keypoints2d:
        if confidence > th:
            cv2.circle(debug_image, (int(x), int(y)), radius=1, color=(0,255,0), thickness=1)
    return debug_image


class DebugMonitor:

    def __init__(self):
        self.isActive = False
        self.window = None
        self.client = None
        self.buffer = {}

    def open(self, udp_port=None):
        if udp_port is not None:
            self.client = UDPClient(host='', port=udp_port)
        layout = [
            [sg.Multiline('', size=(80,30), expand_x=True, expand_y=True, key='-OUT-')],
            [sg.Stretch(), sg.Button('Clear', key='-CLEAR-')]
            ]
        self.window = sg.Window(title='Debug Monitor', layout=layout, finalize=True, resizable=True,return_keyboard_events=True)
        self.isActive = True
        return
    
    def close(self):
        if not self.isActive:
            return
        if self.client is not None:
            self.client.close()
        self.window.close()
        self.isActive = False

    def update(self):
        if not self.isActive:
            return 
        data = None
        if self.client is not None:
            data = self.client.listen()
        if data:
            json_data = json.loads(data)
            self.add_line(json_data['timestamp'])
        else:
            pass
        event, values = self.window.read(timeout=0)
        if event == '-CLEAR-':
            self.clear()
        if event is None:
            self.close()
        return 

    def add_line(self, text):
        if not self.isActive:
            return
        self.window['-OUT-'].print(text)
        return

    def write(self, t, keypoints3d):
        if keypoints3d is not None:
            self.buffer[t] = keypoints3d.tolist()

    def out(self):
        with open('output.json', 'w') as f:
            json.dump(self.buffer, f, indent=4)

    def clear(self):
        self.window['-OUT-'].update('')
        return


if __name__ == '__main__':

    monitor = DebugMonitor()
    monitor.open(udp_port=5555)

    while True:
        event, values = monitor.update()

        if event is None:
            break

    monitor.close()