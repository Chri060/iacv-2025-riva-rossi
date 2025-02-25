class Camera():
    def __init__(self,name, intrinsic, extrinsic, distortion, position, rotation):
        self.name = name
        self.intrinsic = intrinsic # 3x3 matrix [K]
        self.extrinsic = extrinsic # 3x4 matrix [R|t]
        self.dist = distortion
        self.position = position
        self.rotation = rotation

    def __str__(self):
        return f"""---------------------------------------------\nCamera : {self.name}:\n> Intrinsic:\n{self.intrinsic}\n> Distortion:\n{self.dist}\n> Position:\n{self.position}\n> Rotation:\n{self.rotation}\n---------------------------------------------"""