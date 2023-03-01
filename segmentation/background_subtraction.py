'''
Background subtraction program for human segmentation from image

variables:
    background : background image
    a_min : threshold of brightness distortion between image and background (calculate at each pixel)
    c_max : threshold of color distortion between image and background (calculate at each pixel)

function:
    set_background() : set background image
    process() : input image, output segmented mask (1 for foreground, 0 for background)
'''


import cv2
import numpy as np
import PySimpleGUI as sg
sg.theme('DarkBlue')

from segmentation.setting import SegmentationMethod
from tools.preprocess import rgb_to_gray


class BackgroundSubtractor:
    def __init__(self, background=None, img_path='', a_min=0.7, a_max=0.95, c_min=0, c_max=30):
        self.type = SegmentationMethod.Subtraction
        self.background = background
        self.img_path = img_path
        self.a_min = a_min
        self.a_max = a_max
        self.c_min = c_min
        self.c_max = c_max

        if img_path != '':
            self.load_background(img_path)

    def load_config(self, config):
        self.a_min = config['a_min']
        self.a_max = config['a_max']
        self.c_min = config['c_min']
        self.c_max = config['c_max']
        if config['background_path'] != '':
            self.load_background(config['background_path'])
        

    def set_background(self, background):
        self.background = background

    def load_background(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.set_background(image)
        self.img_path = img_path

    def process(self, image):
        if self.background is None:
            return np.ones(image.shape[:2]+(1,))
        E = self.background.astype(np.int64)
        I = image.astype(np.int64)
        a = (E*I).sum(axis=-1) / np.maximum((E*E).sum(axis=-1), 1e-5)
        c = np.linalg.norm(I - a[:,:,None]*E, axis=-1)
        cond = (a < self.a_min) | (c > self.c_max)
        mask = np.where(cond, 1, 0)
        mask = cv2.medianBlur(mask.astype(np.float32), 5)
        mask = np.expand_dims(mask, axis=-1)
        return mask

    # 旧型式
    def apply(self, image):
        if self.background is None:
            return None
        # グレースケール化
        image_gray = rgb_to_gray(image[None,:,:,:])[0,:,:,0]
        background_gray = rgb_to_gray(self.background[None,:,:,:])[0,:,:,0]

        # 影の検出
        E = self.background.astype(np.int64)
        I = image.astype(np.int64)
        a = (E*I).sum(axis=-1) / np.maximum((E*E).sum(axis=-1), 1e-5)
        c = np.linalg.norm(I - a[:,:,None]*E, axis=-1)
        cond = (self.a_min <= a) & (a <= self.a_max) & (c <= self.c_max)
        shadow_mask = np.where(cond,0,1)
        shadow_mask = cv2.medianBlur(shadow_mask.astype(np.float32), 5)

        # 背景差分
        diff = np.abs(image_gray - background_gray)
        # マスク画像の作成
        mask = np.where(diff>15, 1, 0).astype(np.float32)
        # メディアンフィルタでノイズ除去
        mask = cv2.medianBlur(mask, 5)

        mask = (mask * shadow_mask)[:,:,None]
        foreground = (image * mask).astype(np.uint8)
        return foreground
    
    
def setup_subtractor(camera, subtractor):
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
            break
        if event is None or event == '-Cancel-':
            break
    window.close()
    return True




if __name__ == '__main__':
    from camera.camera import USB_Camera
    camera = USB_Camera('cam', device_id=0)
    camera.open()

    subtractor = BackgroundSubtractor()

    while True:
        img = camera.get_image()

        mask = subtractor.process(img)
        if mask is not None:
            debug_image = (img * mask).astype(np.uint8)
        else:
            debug_image = img
        cv2.imshow(camera.name, debug_image)

        key = cv2.waitKey(1)
        if key == ord('s'):
            subtractor.set_background(img)
        if key == 27:
            break

    camera.close()
