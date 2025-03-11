import cv2 as cv
import pickle
import os
import random
import numpy as np


class Environment:
    video_names = []
    camera_names = []
    paths = {}
    env_vars = {}
    savename = None
    __initialized = False

    @staticmethod
    def start_pipeline(pipeline_configs):
        import pipeline.processors.calibration as cal
        import pipeline.processors.video_processing as vp
        import pipeline.processors.localization as loc
        import pipeline.visualizers.viz as viz

        pipes = {
            "intrinsic": cal.Intrinsic_Calibration,
            "video_synchronization": vp.Synchronizer,
            "video_stabilization": vp.Stabilizer,
            "video_undistortion": vp.Undistorcer,
            "lane_detection": loc.Lane_Detector,
            "extrinsic": cal.Extrinsic_Calibration,
            "ball_tracker": loc.Ball_Tracker,
            "ball_localization": loc.Ball_Localization,
            "localization_viz": viz.Camera_Localization_Viz,
            "ball_localization_viz": viz.Ball_Localization_Viz
        }

        # Initialize the pipes with the given savename
        for key in pipes.keys():
            pipes.update({key: pipes.get(key)(Environment.savename)})

        # Process each pipe
        for proc_conf in pipeline_configs:
            proc = pipes[proc_conf["name"]]
            params = proc_conf.get("params", None)
            proc_type = proc_conf["type"]
            print(
                f"\n>>>>> {proc.__class__.__name__} | {proc_type} <<<<<\n       params : {params}\n"
            )
            if proc_type == "execute":
                proc.execute(params)
            elif proc_type == "load":
                proc.load(params)

    @staticmethod
    def initialize_globals(savename, global_parameters):
        print(f"Saves as : {savename}")
        Environment.savename = savename
        Environment.video_names = global_parameters.get("video_names", [])
        Environment.camera_names = global_parameters.get("camera_names", [])
        Environment.paths = global_parameters.get("paths", {})
        Environment.env_vars = {}
        Environment.coords = global_parameters.get("coords", {})
        Environment.visualization = global_parameters.get("visualization", False)
        print("Default visualization : ", Environment.visualization)
        Environment.__initialized = True
        # Load the videos
        Environment.__load_views()

    @staticmethod
    def get_views():
        return [Environment.get(name) for name in Environment.camera_names]

    @staticmethod
    def get(name):
        if not Environment.__initialized:
            raise Exception("Environment has not been initialized")
        if name in Environment.env_vars:
            return Environment.env_vars.get(name)
        raise Exception(
            f'Trying to access "{name}" in Environment while it does not exists'
        )

    @staticmethod
    def set(name, obj, overwrite=False):
        if not Environment.__initialized:
            raise Exception("Environment has not been initialized")
        if not overwrite and name in Environment.env_vars:
            raise Exception("Trying to overwrite without permission")
        Environment.env_vars[name] = obj

    @staticmethod
    def __load_views():
        video_names = Environment.video_names
        camera_names = Environment.camera_names
        originals_path = Environment.paths["originals"]
        for i, camera_name in enumerate(camera_names):
            full_path = f"{originals_path}/{camera_name}/{video_names[i]}"
            video = Video(full_path)
            camera = Camera(camera_name)
            lane = Lane()
            trajectory = None
            view = View(camera, video, lane, trajectory)
            Environment.set(camera_name, view)


class DataManager:
    save_path = "resources/pickle/"

    @staticmethod
    def save(obj, savename):
        with open(f"{DataManager.save_path}{savename}.pkl", "wb") as out:
            pickle.dump(obj, out, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(name):
        try:
            with open(f"{DataManager.save_path}{name}.pkl", "rb") as inp:
                obj = pickle.load(inp)
                return obj
        except FileNotFoundError:
            raise Exception(f"Failed to find load : {name}")

    @staticmethod
    def delete(name):
        print(f"Deleting {name}")
        os.remove(f"{DataManager.save_path}{name}.pkl")
        print(f"> {name} deleted")


class Camera:
    def __init__(
        self,
        name,
        intrinsic=None,
        distortion=None,
        extrinsic=None,
        position=None,
        rotation=None,
    ):
        self.name = name
        self.intrinsic = intrinsic
        self.distortion = distortion
        self.extrinsic = extrinsic
        self.position = position
        self.rotation = rotation

    def __str__(self):
        return f"""---------------------------------------------\nCamera : {self.name}:\n> Intrinsic:\n{self.intrinsic}\n> Distortion:\n{self.distortion}\n> Position:\n{self.position}\n> Rotation:\n{self.rotation}\n---------------------------------------------"""


class Video:
    def __init__(self, path):
        self.capture = cv.VideoCapture(path)  # opencv video capture
        self.path = path  # textual path to the saved video

    def get_video_properties(self):
        fps = self.capture.get(cv.CAP_PROP_FPS)
        frame_count = int(self.capture.get(cv.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps else 0
        width = int(self.capture.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        return fps, duration, (width, height)


class Lane:
    def __init__(self, corners=None):
        self.corners = corners


class Ball_Trajectory_2D:
    def __init__(self, n_frames, fps=None, image_points=None, radiuses=None):
        self.n_frames = n_frames
        self.fps = fps
        if image_points is not None: 
            assert len(image_points) == n_frames and len(radiuses) == n_frames
        self.image_points = image_points or [None] * n_frames # empty list if no image_points are provided
        self.radiuses = radiuses or [None] * n_frames
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    def set_by_frame(self, coord, r, curr_frame):
        if curr_frame > self.n_frames : 
            raise Exception("Trying to access an out of bount frame")
        self.image_points[curr_frame] = coord
        self.radiuses[curr_frame] = r

    def get_by_frame(self, curr_frame):
        if curr_frame > self.n_frames : 
            raise Exception("Trying to access an out of bount frame")
        return self.image_points[curr_frame], self.radiuses[curr_frame]
    
    def plot_onto(self, image):
        to_plot = []
        for curr_frame, curr_pos in enumerate(self.image_points):
            curr_rad = self.radiuses[curr_frame]
            if curr_pos is not None:
                curr_pos = (int(curr_pos[0]), int(curr_pos[1]))
                curr_rad = int(curr_rad)
                to_plot.append(curr_pos)
                cv.circle(image, curr_pos, curr_rad, self.color, 1)
        to_plot = np.array(to_plot, dtype=np.int32).reshape((-1, 1, 2))
        cv.polylines(image, to_plot, isClosed=False, color=self.color, thickness=2)
            


class View:
    def __init__(self, camera, video, lane, trajectory):
        self.camera = camera
        self.video = video
        self.lane = lane
        self.trajectory = trajectory
