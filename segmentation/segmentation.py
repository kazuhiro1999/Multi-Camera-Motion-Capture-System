import onnxruntime
import cv2
import numpy as np
import mediapipe as mp
from segmentation.setting import SegmentationMethod
from tools.preprocess import crop_or_pad


class PaddleSegmentation:
    model_path = 'segmentation/models/ppmattingv2.onnx'

    def __init__(self):
        self.type = SegmentationMethod.Paddle
        self.session = Session(self.model_path, executeType='CPUExecutionProvider')
        self.input_size = (512,512)

    def process(self, image): 
        image_height, image_width = image.shape [:2]     
        input_image = cv2.resize(image, dsize=self.input_size) # resize
        inputs = self.preprocess_image(input_image)
        outputs = self.session.execute(inputs)
        mask = self.detect_mask(outputs)
        mask = cv2.resize(mask, dsize=(image_width, image_height))
        mask = np.expand_dims(mask, axis=-1)
        return mask

    def preprocess_image(self, input_image):
        inputs = (input_image / 255 - [0.5,0.5,0.5]) / [0.5,0.5,0.5] # normalize
        inputs = inputs.transpose(2,0,1).astype('float32')
        inputs = np.expand_dims(inputs, axis=0)
        return inputs

    def detect_mask(self, output):
        out = np.squeeze(output[0])
        mask = np.where(out > 0.5, 1, 0)
        return mask.astype(np.uint8)


class DeepLabV3:
    model_path = 'segmentation/models/DeepLabV3.onnx'

    def __init__(self):
        self.type = SegmentationMethod.DeepLabV3
        self.session = Session(self.model_path, executeType='CPUExecutionProvider')
        self.input_size = (224,224)

    def process(self, image):
        image_height, image_width = image.shape[:2]
        input_image = cv2.resize(image, dsize=self.input_size)
        inputs = self.preprocess_image(input_image)
        outputs = self.session.execute(inputs)
        mask = self.detect_mask(outputs)
        mask = cv2.resize(mask, dsize=(image_width, image_height))
        mask = np.expand_dims(mask, axis=-1)
        return mask

    def preprocess_image(self, input_image):
        inputs = (input_image / 255 - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
        inputs = inputs.transpose(2,0,1).astype('float32')
        inputs = np.expand_dims(inputs, axis=0)
        return inputs

    def detect_mask(self, output):
        out = np.squeeze(output[0])
        out = np.argmax(out, axis=0)
        mask = np.where(out==15, 1, 0)
        return mask[:,:,None]


class MediaPipeSelfieSegmentation:

    def __init__(self):
        self.type = SegmentationMethod.Mediapipe
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

    def process(self, image):
        results = self.selfie_segmentation.process(image)
        mask = (results.segmentation_mask > 0.1)[:,:,None]
        return mask


class Session:
    def __init__(self, model_path='models/onnx/DeepLabV3.onnx', executeType='CPUExecutionProvider'):
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
    camera.open(device_id=0)

    segmentation = MediaPipeSelfieSegmentation()

    while True:
        image = camera.get_image()
        
        #image = crop_or_pad(image)
        mask = segmentation.process(image)
        debug_image = (image * mask).astype(np.uint8)
        cv2.imshow('mask', debug_image)

        if cv2.waitKey(1) == 27:
            break
