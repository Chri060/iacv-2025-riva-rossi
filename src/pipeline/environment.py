import os
import pickle
import random

import cv2 as cv
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray
from scipy.interpolate import Akima1DInterpolator

from pipeline.pipe import Pipe


class Camera:
    def __init__(
        self,
        name: str,
    ):
        self.name = name
        self.intrinsic: NDArray | None = None
        self.distortion: NDArray | None = None
        self.extrinsic: NDArray | None = None
        self.position: NDArray | None = None
        self.rotation: NDArray | None = None

    def __str__(self):
        return f"""---------------------------------------------\nCamera : {self.name}:\n> Intrinsic:\n{self.intrinsic}\n> Distortion:\n{self.distortion}\n> Position:\n{self.position}\n> Rotation:\n{self.rotation}\n---------------------------------------------"""


class Video:
    def __init__(self, path: str):
        self.capture = cv.VideoCapture(path)  # opencv video capture
        self.path = path  # textual path to the saved video

    def get_video_properties(self) -> tuple[float, float, tuple[int]]:
        """
        Returns video properties.

        Returns:
            tuple: A tuple containing:
                - fps (float): Frames per second.
                - duration (float): Duration in seconds.
                - (width, height) (tuple[int, int]): Resolution of the video.
        """
        fps = self.capture.get(cv.CAP_PROP_FPS)
        frame_count = int(self.capture.get(cv.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps else 0
        width = int(self.capture.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        return fps, duration, (width, height)


class Lane:
    def __init__(self, corners: NDArray | None = None):
        self.corners: NDArray | None = corners


class Ball_Trajectory_2D:
    def __init__(
        self,
        n_frames: int,
        fps: float | None = None,
        image_points: NDArray | None = None,
        radiuses: NDArray | None = None,
        start: int | None = None,
        end: int | None = None,
    ):
        self.n_frames: int = n_frames
        self.fps: float = fps
        self.start: int = start
        self.end: int = end

        if image_points is not None:
            assert len(image_points) == n_frames and len(radiuses) == n_frames

        self.image_points = image_points or np.array(
            [[None, None]] * n_frames
        )  # empty list if no image_points are provided

        self.radiuses = radiuses or np.array([None] * n_frames)

        self.color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )

    def set_by_frame(self, coord: NDArray, r: float, curr_frame: int) -> None:
        if curr_frame > self.n_frames or curr_frame < 0:
            raise Exception("Trying to access an out of bount frame")
        self.image_points[curr_frame] = coord
        self.radiuses[curr_frame] = r

        if self.start is None:
            self.start = curr_frame
            self.end = curr_frame
        elif curr_frame < self.start:
            self.start = curr_frame
        elif curr_frame > self.end:
            self.end = curr_frame

    def get_by_frame(self, curr_frame: int) -> tuple[NDArray, float]:
        """
        Returns:
            tuple: Containing the information at that frame:
                - coord (tuple[float]) : coordinates in the image reference frame
                - radius (float) : radius size in the image reference frame
        """
        if curr_frame > self.n_frames:
            raise Exception("Trying to access an out of bound frame")
        return self.image_points[curr_frame], self.radiuses[curr_frame]

    def get_coords(self, start: int | None = None, end: int | None = None) -> NDArray:
        """
        Returns:
            NDArray: The coordinates in the [self.start:self.end] range or the specified one
        """
        return self.image_points[start or self.start : end or self.end]

    def get_radiuses(self, start: int | None = None, end: int | None = None) -> NDArray:
        """
        Returns:
            NDArray: The radiuses in the [self.start:self.end] range or the specified one
        """
        return self.radiuses[start or self.start : end or self.end]

    @staticmethod
    def interpolate_array(arr, window=8, inverse_fit=False):
        """
        Interpolates/extrapolates None values in arr using linear fit over a sliding window.
        Keeps initial None values untouched. Converts interpolated numbers to int.
        :param arr: list of values (floats or None)
        :param window: number of past points to consider for extrapolation
        :param inverse_fit: if True, fit 1/y = a*x + b
        """
        n = len(arr)
        arr_interp = arr.copy()

        # Find first non-None index
        first_known = next((i for i, v in enumerate(arr) if v is not None), None)

        if first_known is None:
            return arr_interp  # all None, nothing to do

        # Only interpolate/extrapolate after first known value
        for i in range(first_known + 1, n):
            if arr_interp[i] is None:
                # Collect last 'window' known points before i
                known_indices = [j for j in range(max(first_known, i - window), i) if arr_interp[j] is not None]
                if len(known_indices) >= 2:
                    x_known = np.array(known_indices)
                    y_known = np.array([arr_interp[j] for j in known_indices], dtype=float)

                    if inverse_fit:
                        y_inv = 1 / y_known
                        coeffs = np.polyfit(x_known, y_inv, 1)
                        arr_interp[i] = int(round(1 / np.polyval(coeffs, i)))
                    else:
                        coeffs = np.polyfit(x_known, y_known, 1)
                        arr_interp[i] = int(round(np.polyval(coeffs, i)))
                elif known_indices:
                    arr_interp[i] = int(arr_interp[known_indices[-1]])
                else:
                    next_known = next((arr_interp[j] for j in range(i + 1, n) if arr_interp[j] is not None), None)
                    if next_known is not None:
                        arr_interp[i] = int(next_known)

        # Convert all known values after first_known to int
        for i in range(first_known, n):
            if arr_interp[i] is not None:
                arr_interp[i] = int(arr_interp[i])

        return arr_interp

    def interpolate_centers_2d(self, window=8):
        """
        Interpolates missing 2D points (x, y) together using linear fit over a sliding window.
        Keeps initial None points untouched and converts all interpolated points to int.
        """
        n = len(self.image_points)
        interp_points = self.image_points.copy()

        # Find first non-None point
        first_known = next((i for i, p in enumerate(interp_points) if p is not None and all(v is not None for v in p)),
                           None)
        if first_known is None:
            return  # all None, nothing to do

        for i in range(first_known + 1, n):
            if interp_points[i] is None or np.any(interp_points[i] == None):
                # Collect last 'window' known points before i
                known_indices = [j for j in range(max(first_known, i - window), i) if interp_points[j] is not None]
                if len(known_indices) >= 2:
                    t_known = np.array(known_indices)
                    x_known = np.array([interp_points[j][0] for j in known_indices], dtype=float)
                    y_known = np.array([interp_points[j][1] for j in known_indices], dtype=float)

                    # Linear fit x(t) and y(t) together
                    A = np.vstack([t_known, np.ones(len(t_known))]).T
                    x_coeff = np.linalg.lstsq(A, x_known, rcond=None)[0]
                    y_coeff = np.linalg.lstsq(A, y_known, rcond=None)[0]

                    interp_points[i] = [int(round(np.dot([i, 1], x_coeff))),
                                        int(round(np.dot([i, 1], y_coeff)))]
                elif known_indices:
                    interp_points[i] = [int(interp_points[known_indices[-1]][0]),
                                        int(interp_points[known_indices[-1]][1])]
                else:
                    next_known = next((interp_points[j] for j in range(i + 1, n) if interp_points[j] is not None), None)
                    if next_known is not None:
                        interp_points[i] = [int(next_known[0]), int(next_known[1])]

        # Convert all known points after first_known to int
        for i in range(first_known, n):
            if interp_points[i] is not None:
                interp_points[i] = [int(interp_points[i][0]), int(interp_points[i][1])]

        self.image_points = interp_points

    def interpolate_all(self):
        # Radiuses: inverse fit for perspective scaling
        self.radiuses = self.interpolate_array(self.radiuses, window=10, inverse_fit=True)

        # Centers: interpolate x and y together
        self.interpolate_centers_2d(window=8)

    def plot_onto(self, image: MatLike) -> None:
        """
        Plots the trajectory as a series of circles on an image.
        Skips frames where coordinates are None or NaN.
        """
        to_plot = []
        for curr_frame, curr_pos in enumerate(self.image_points):
            curr_rad = self.radiuses[curr_frame]

            # Skip if coordinates are None or NaN
            if curr_pos[0] is None or curr_pos[1] is None:
                continue
            if np.isnan(curr_pos[0]) or np.isnan(curr_pos[1]):
                continue

            cx, cy = map(int, curr_pos)

            # Draw circle if radius is valid
            if curr_rad is not None and not np.isnan(curr_rad):
                cv.circle(image, (cx, cy), int(curr_rad), self.color, 1)

            to_plot.append((cx, cy))

        if to_plot:
            to_plot = np.array(to_plot, dtype=np.int32).reshape((-1, 1, 2))
            cv.polylines(image, [to_plot], isClosed=False, color=self.color, thickness=2)


class Ball_Trajectory_3D:
    def __init__(
        self,
        n_frames: int,
        fps: int | None = None,
        coords: NDArray | None = None,
        radius: int | None = None,
        start: int | None = None,
        end: int | None = None,
    ):
        self.n_frames = n_frames
        self.start = start
        self.end = end
        self.fps = fps
        if coords is not None:
            assert len(coords) == n_frames
        self.coords = coords or np.array(
            [[None, None, None]] * n_frames
        )  # empty list if no image_points are provided
        self.radius = radius
        self.color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )

    def set_by_frame(self, coord: NDArray, curr_frame: int) -> None:
        if curr_frame > self.n_frames:
            raise Exception("Trying to access an out of bount frame")
        self.coords[curr_frame] = coord
        if self.start is None:
            self.start = curr_frame
            self.end = curr_frame
        elif curr_frame < self.start:
            self.start = curr_frame
        elif curr_frame > self.end:
            self.end = curr_frame

    def get_by_frame(self, curr_frame: int) -> NDArray:
        if curr_frame > self.n_frames:
            raise Exception("Trying to access an out of bount frame")
        return self.coords[curr_frame]

    def get_coords(self, start: int | None = None, end: int | None = None):
        start = start or self.start
        end = end or self.end
        return self.coords[start:end]


class View:
    def __init__(
        self,
        camera: Camera,
        video: Video,
        lane: Lane | None = None,
        trajectory: Ball_Trajectory_2D | None = None,
    ):
        self.camera = camera
        self.video = video
        self.lane = lane or Lane()
        self.trajectory = trajectory


class Environment:
    CV_VISUALIZATION_NAME = "OpenCV Visualization"
    video_names: list[str] = []
    camera_names: list[str] = []
    paths: dict[str, str] = {}
    env_vars: dict[str, object] = {}
    savename: str = None
    __initialized: bool = False

    @staticmethod
    def initialize_globals(savename: str, global_parameters: dict) -> None:
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
    def __load_views() -> None:
        """
        Views initialization after the global configs are passed to the environment
        """
        video_names = Environment.video_names
        camera_names = Environment.camera_names
        originals_path = Environment.paths["originals"]
        for i, camera_name in enumerate(camera_names):
            full_path = f"{originals_path}/{camera_name}/{video_names[i]}"
            video = Video(full_path)
            camera = Camera(camera_name)
            view = View(camera, video)
            Environment.set(camera_name, view)

    @staticmethod
    def start_pipeline(pipeline_configs: dict) -> None:
        import pipeline.pipes.calibration as cal
        import pipeline.pipes.localization as loc
        import pipeline.pipes.video_processing as vp

        pipes: dict[str, Pipe] = {
            "intrinsic": cal.IntrinsicCalibration,
            "video_synchronization": vp.SynchronizeVideo,
            "video_undistortion": vp.UndistortVideo,
            "lane_detection": loc.Lane_Detector,
            "extrinsic": cal.ExtrinsicCalibration,
            "ball_tracker_hough": loc.Ball_Tracker_Hough,
            "ball_tracker_yolo": loc.Ball_Tracker_YOLO,
            "ball_localization": loc.Ball_Localization,
        }

        # Initialize the pipes with the given savename
        for key in pipes.keys():
            pipes.update({key: pipes.get(key)(Environment.savename)})

        # Process each pipe
        for pipe_conf in pipeline_configs:
            pipe = pipes[pipe_conf["name"]]
            params: dict = pipe_conf.get("params", None)
            proc_type: str = pipe_conf["type"]
            print(
                f"\n>>>>> {pipe.__class__.__name__} | {proc_type} <<<<<\n       params : {params}\n"
            )
            if proc_type == "execute":
                pipe.execute(params)
            elif proc_type == "load":
                if pipe_conf["name"] == "intrinsic": 
                    pipe.load()
                else: 
                    pipe.load(params)

        cv.destroyAllWindows()

    @staticmethod
    def get(name: str) -> object:
        """
        Returns:
            object: The object stored as an environment variable under the specified name.
        """
        if not Environment.__initialized:
            raise Exception("Environment has not been initialized")
        if name in Environment.env_vars:
            return Environment.env_vars.get(name)
        raise Exception(
            f'Trying to access "{name}" in Environment while it does not exists'
        )

    @staticmethod
    def set(name: str, obj: object, overwrite: bool = False) -> None:
        if not Environment.__initialized:
            raise Exception("Environment has not been initialized")
        if not overwrite and name in Environment.env_vars:
            raise Exception("Trying to overwrite without permission")
        Environment.env_vars[name] = obj

    @staticmethod
    def get_views() -> list[View]:
        return [Environment.get(name) for name in Environment.camera_names]


class DataManager:
    save_path: str = "resources/pickle/"

    @staticmethod
    def save(obj: object, savename: str) -> None:
        with open(f"{DataManager.save_path}{savename}.pkl", "wb") as out:
            pickle.dump(obj, out, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(savename: str) -> None:
        try:
            with open(f"{DataManager.save_path}{savename}.pkl", "rb") as inp:
                obj = pickle.load(inp)
                return obj
        except FileNotFoundError:
            raise Exception(f"Failed to find load : {savename}")

    @staticmethod
    def delete(savename: str) -> None:
        print(f"Deleting {savename}")
        os.remove(f"{DataManager.save_path}{savename}.pkl")
        print(f"> {savename} deleted")
