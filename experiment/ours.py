import cv2
import numpy as np
import onnxruntime
import time

from tools.preprocess import crop_or_pad

class Session:
    def __init__(self, model_path='pose/models/model_pose2d_mobile_rgb_sharp.onnx', executeType='CUDAExecutionProvider'):
        self.session = onnxruntime.InferenceSession(model_path, providers=[executeType])
        self.input_name = self.session.get_inputs()[0].name
        self.outputs = [output.name for output in self.session.get_outputs()]
        
    def execute(self, x):
        inputs = {self.input_name : x}
        result = self.session.run(self.outputs, inputs)
        return result

if __name__ == '__main__':
    cap = cv2.VideoCapture("data/0_keito_1_1.mp4")
    session = Session()
    t_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image = frame.copy()
        image = crop_or_pad(image)
        image = cv2.resize(image, dsize=(224,224))
        image = np.expand_dims(image, axis=0).astype(np.float32) / 255
        t0 = time.time()
        out = session.execute(image)
        t_list.append(time.time() - t0)

        cv2.imshow('image', frame)

        if cv2.waitKey(1) == 0:
            break

        if len(t_list) > 1000:
            break

    print(np.array(t_list).mean())