import cv2
import numpy as np

def crop_or_pad(image):
    image_height, image_width = image.shape[:2]
    if image_height > image_width: # 縦画像はサイドを0パディング
        pad = (image_height - image_width) // 2
        img = cv2.copyMakeBorder(image, 0, 0, pad, pad, cv2.BORDER_CONSTANT, (255,255,255))
    elif image_width > image_height: # 横画像はセンタークロップ
        x_min = (image_width - image_height) // 2
        x_max = (image_width + image_height) // 2
        img = image[:,x_min:x_max]
    else:
        img = image.copy()
    return img



def rgb_to_gray(image):
    assert (image.ndim == 3 or image.ndim == 4)
    if image.ndim == 4:
        gray = 0.299*image[:,:,:,0] + 0.587*image[:,:,:,1] + 0.114*image[:,:,:,2]
        return gray[:,:,:,None] 
    elif image.ndim == 3:
        gray = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
        return gray[:,:,None] 
    else:
        return None