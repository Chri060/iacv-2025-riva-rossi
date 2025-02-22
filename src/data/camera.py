class Camera():
    def __init__(self, intrinsic, extrinsic, distortion):
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.dist = distortion