from enum import Enum

# カメラの種類
class CameraType(Enum):
    none = 0
    USB = 1
    M5 = 2
    Video = 3


class SegmentationMethod(Enum):
    none = 0,
    Subtraction = 1,
    Mediapipe = 2,
    Paddle = 3,
    DeepLabV3 = 4


class ModelType(Enum):
    none = 0,
    MediapipePose = 1,
    Humanoid = 2,
    MediapipeHolistic = 3

    @staticmethod
    def get_num_of_joints(model_type):
        if model_type == ModelType.none:
            return 0
        elif model_type == ModelType.MediapipePose:
            return 33      
        elif model_type == ModelType.Humanoid:
            return 38        
        elif model_type == ModelType.MediapipeHolistic:
            return 75
        else:
            return 0