from enum import Enum


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
