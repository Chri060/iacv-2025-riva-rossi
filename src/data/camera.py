class Camera():
    def __init__(self, intrinsic, extrinsic, distortion):
        self.intrinsic = intrinsic # 3x3 matrix [K]
        self.extrinsic = extrinsic # 3x4 matrix [R|t]
        self.dist = distortion