import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time
from tools.visualization import draw_keypoints

def MoveNet():
    model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
    input_size = 256

    module = hub.load(model_url)
    model = module.signatures["serving_default"]
    return model


if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    model = MoveNet()
    t_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        input_image = tf.expand_dims(frame, axis=0)
        input_image = tf.image.resize_with_pad(input_image, 256, 256)
        input_image = tf.cast(input_image, dtype=tf.int32)
        # Run model inference.
        t0 = time.time()
        outputs = model(input_image)
        t_list.append(time.time() - t0)
        # Output is a [1, 1, 17, 3] tensor.
        keypoints_with_scores = outputs['output_0'].numpy()[0,0]
        
        debug_image = draw_keypoints(frame, keypoints_with_scores)
        cv2.imshow('debug_image', debug_image)
        
        if cv2.waitKey(1) == 27:
            break

        if len(t_list) > 100:
            break

    print(np.array(t_list).mean())
        