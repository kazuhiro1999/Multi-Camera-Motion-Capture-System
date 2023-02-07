from enum import Enum

class SegmentationMethod(Enum):
    none = 0,
    Subtraction = 1,
    Mediapipe = 2,
    Paddle = 3,
    DeepLabV3 = 4