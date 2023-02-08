'''
Our 2D pose estimation model (estimate pose from segmented RGB image)

function:
    process() : input RGB image, output 2D coordinate of each key-points
'''


import onnxruntime
import cv2
import numpy as np

from pose.setting import ModelType
from segmentation.segmentation import *
from tools.preprocess import crop_or_pad
from tools.visualization import draw_keypoints


class PoseNet:

    KEYPOINT_DICT = {
        'Hips':0,
        'Spine':1,
        'Chest':2,
        'UpperChest':3,
        'Neck':4,
        'HeadTop':5,
        'Nose':6,
        'L_Eye':7,
        'R_Eye':8,
        'L_Ear':9,
        'R_Ear':10,
        'Jaw':11,
        'L_Shoulder':12,
        'L_UpperArm':13,
        'L_LowerArm':14,
        'R_Shoulder':15,
        'R_UpperArm':16,
        'R_LowerArm':17,
        'L_UpperLeg':18,
        'L_LowerLeg':19,
        'R_UpperLeg':20,
        'R_LowerLeg':21,
        'L_Hand':22,
        'L_Palm':23,
        'L_Thumb':24,
        'L_FingerTip':25,
        'L_Little':26,
        'R_Hand':27,
        'R_Palm':28,
        'R_Thumb':29,
        'R_FingerTip':30,
        'R_Little':31,
        'L_Foot':32,
        'L_Heel':33,
        'L_Toe':34,
        'R_Foot':35,
        'R_Heel':36,
        'R_Toe':37,
    }

    def __init__(self, model_path='pose/models/model_pose2d_mobile_rgb_sharp.onnx'):
        self.type = ModelType.Ours
        self.session = Session(model_path, executeType='CPUExecutionProvider')
        self.input_size = (224,224)

    def process(self, image):       
        image_height, image_width = image.shape[:2] 
        input_image = crop_or_pad(image) # center crop
        input_image = cv2.resize(input_image, dsize=self.input_size) # resize
        inputs = self.preprocess_image(input_image)
        outputs = self.session.execute(inputs)
        keypoints = self.detect_keypoints(outputs[0])
        keypoints2d = self.reshape_back(keypoints, image_height, image_width)
        return keypoints2d

    def preprocess_image(self, input_image):
        inputs = input_image.astype(np.float32) / 255 # normalize
        inputs = np.expand_dims(inputs, axis=0)
        return inputs

    def detect_keypoints(self, batch_heatmaps):
        batch_size, height, width, num_joints = batch_heatmaps.shape
        heatmaps_reshaped = batch_heatmaps.reshape((batch_size, -1, num_joints))
        idx = np.argmax(heatmaps_reshaped, axis=1).reshape((batch_size, num_joints, 1))
        maxvals = np.amax(heatmaps_reshaped, axis=1).reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)
        preds *= pred_mask

        batch_keypoints2d = np.concatenate([preds*4, maxvals], axis=-1)
        return batch_keypoints2d[0]

    def reshape_back(self, keypoints, image_height, image_width):
        keypoints2d = keypoints.copy()
        if image_height > image_width: # 縦画像
            keypoints2d[:,:2] *= (image_height / self.input_size[0])
            pad = (image_height - image_width) // 2
            keypoints2d[:,0] += pad
        elif image_width > image_height: # 横画像
            keypoints2d[:,:2] *= (image_height / self.input_size[0])
            pad = (image_width - image_height) // 2
            keypoints2d[:,0] += pad
        else:
            pass
        return keypoints2d


class Session:
    def __init__(self, model_path='pose/models/model_pose2d_mobile_rgb_sharp.onnx', executeType='CPUExecutionProvider'):
        self.session = onnxruntime.InferenceSession(model_path, providers=[executeType])
        self.input_name = self.session.get_inputs()[0].name
        self.outputs = [output.name for output in self.session.get_outputs()]
        
    def execute(self, x):
        inputs = {self.input_name : x}
        result = self.session.run(self.outputs, inputs)
        return result



if __name__ == '__main__':
    from camera.camera import USB_Camera
    camera = USB_Camera('cam 0')
    camera.open(0)

    segmentation = MediaPipeSelfieSegmentation()

    model_path = 'pose/models/model_pose2d_mobile_rgb_sharp.onnx'
    pose_net = PoseNet(model_path=model_path)

    while True:
        image = camera.get_image()
        mask = segmentation.process(image)
        input_image = (image * mask).astype(np.uint8)
        keypoints2d = pose_net.process(input_image)

        debug_image = draw_keypoints(input_image, keypoints2d)
        cv2.imshow(camera.name, debug_image)

        if cv2.waitKey(1) == 27:
            break