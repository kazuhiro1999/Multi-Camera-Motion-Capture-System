
'''
Camera parameter include intrinsic and extrinsic.

Variables:
    image_width : width of the image
    image_height : height of the image
    fov : field of view
    position : 3D coordinate of the camera position
    rotation : 3D coordinate of the camera rotation (euler angles)

function:
    set_intrinsic() : set camera intrinsic parameter (image_width, image_height, fov). default=None means do not update parameter.
    set_transform() : set cameta extrinsic parameter (position, rotation).
    get_camera_matrix() : calculate camera_matrix (K)
    get_projection_matrix() : calculate projection_matrix
'''

import cv2
import numpy as np


class CameraSetting:
    DefaultPosition = np.zeros(3)
    DefaultRotation = np.zeros(3)
    def __init__(self, image_width=0, image_height=0, fov=90, position=DefaultPosition, rotation=DefaultRotation):
        self.image_width = image_width
        self.image_height = image_height
        self.fov = fov
        self.position = position
        self.rotation = rotation

    def set_intrinsic(self, image_width=None, image_height=None, FOV=None):
        if image_height is not None:
            self.image_height = image_height
        if image_width is not None:
            self.image_width = image_width
        if FOV is not None :
            self.fov = FOV
    
    def get_camera_matrix(self):
        if self.image_width is None or self.image_height is None or self.fov is None:
            return None
        focal = (self.image_width/2) / np.tan(np.radians(self.fov/2))
        matrix = np.array([[focal,0,(self.image_width/2)],[0,focal,(self.image_height/2)],[0,0,1]])
        assert matrix.shape == (3,3)
        return matrix

    def set_transform(self, position=None, rotation=None):
        if position is not None:
            assert len(position) == 3
            self.position = np.array(position).flatten()
        if rotation is not None:
            rotation = np.array(rotation, dtype=np.float32)
            if rotation.shape == (3,3):
                rotation = cv2.Rodrigues(rotation)[0]
            assert len(rotation) == 3
            self.rotation = np.array(rotation, dtype=np.float32).flatten()

    def get_projection_matrix(self):
        K = self.get_camera_matrix()
        if K is None or self.position is None or self.rotation is None:
            return None
        R = cv2.Rodrigues(self.rotation)[0]
        t = self.position.reshape([3,1])
        Rc = R.T
        tc = -R.T @ t
        proj_matrix = K.dot(np.concatenate([Rc,tc], axis=1))
        return proj_matrix

