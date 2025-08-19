import os, pickle, random
import cv2 as cv
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray
import pipeline.pipes.calibration as cal
import pipeline.pipes.localization as loc
import pipeline.pipes.video_processing as vp

class Camera:
    """
    Class to store camera parameters and metadata.
    """

    def __init__(self, name: str):
        """
        Initializes a Camera object.

        Parameters:
            name (str): The name or identifier for the camera.
        """

        self.name = name
        self.intrinsic: NDArray | None = None
        self.distortion: NDArray | None = None
        self.extrinsic: NDArray | None = None
        self.position: NDArray | None = None
        self.rotation: NDArray | None = None

    def __str__(self):
        """
        Returns a formatted string showing all camera parameters.
        """

        return f"""---------------------------------------------\nCamera : {self.name}:\n> Intrinsic:\n{self.intrinsic}\n> Distortion:\n{self.distortion}\n> Position:\n{self.position}\n> Rotation:\n{self.rotation}\n---------------------------------------------"""

class Video:
    """
    Class to handle video loading and basic property extraction.
    """

    def __init__(self, path: str):
        """
        Initializes a Video object by opening the video file.

        Parameters:
           path (str): The file path to the video.
        """

        self.capture = cv.VideoCapture(path)
        self.path = path

    def get_video_properties(self) -> tuple[float, float | int, tuple[int, int]]:
        """
        Retrieves basic properties of the video.

        Returns:
            tuple: A tuple containing:
                - fps (float): Frames per second of the video.
                - duration (float): Total duration in seconds.
                - resolution (tuple[int, int]): Width and height of the video in pixels.
        """

        # Get frames per second
        fps = self.capture.get(cv.CAP_PROP_FPS)

        # Get total number of frames
        frame_count = int(self.capture.get(cv.CAP_PROP_FRAME_COUNT))

        # Calculate video duration in seconds
        duration = frame_count / fps if fps else 0

        # Get video resolution
        width = int(self.capture.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv.CAP_PROP_FRAME_HEIGHT))

        return fps, duration, (width, height)

class Lane:
    """
    Class to represent a lane in an image or video frame.
    """

    def __init__(self, corners: NDArray | None = None):
        """
        Initializes a Lane object.

        Parameters:
            corners (NDArray | None): Optional array of corner points defining the lane.
                                      Typically, this could be a set of 2D points in image coordinates
                                      or 3D points in world coordinates.
        """

        self.corners: NDArray | None = corners

class BallTrajectory2d:
    """
    Represents the 2D trajectory of a ball across video frames.
    Stores image coordinates and radius for each frame and allows interpolation and retrieval of trajectory data.
    """

    def __init__(self, n_frames: int, fps: float | None = None, image_points: NDArray | None = None, radii: NDArray | None = None, start: int | None = None, end: int | None = None):
        """
        Initializes the BallTrajectory2d object.

        Args:
           n_frames (int): Number of frames in the video or trajectory.
           fps (float | None): Frames per second (optional).
           image_points (NDArray | None): Array of (x, y) coordinates per frame.
           radii (NDArray | None): Array of radius values per frame.
           start (int | None): Start the frame of the detected trajectory.
           end (int | None): End frame of the detected trajectory.
        """

        self.n_frames: int = n_frames
        self.fps: float = fps
        self.start: int = start
        self.end: int = end

        # Ensure provided arrays have the correct length
        if image_points is not None:
            assert len(image_points) == n_frames and len(radii) == n_frames

        # Initialize image points and radii if not provided
        self.image_points = image_points or np.array([[None, None]] * n_frames)
        self.radii = radii or np.array([None] * n_frames)

        # Assign a random color for visualization
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def set_by_frame(self, coord: NDArray, r: float, curr_frame: int) -> None:
        """
        Set the ball's position and radius for a specific frame.

        Args:
            coord (NDArray): 2D coordinates (x, y) in the image frame.
            r (float): Radius of the ball in the image frame.
            curr_frame (int): Frame index to update.
        """

        if curr_frame > self.n_frames or curr_frame < 0:
            raise Exception("Trying to access an out of bount frame")
        self.image_points[curr_frame] = coord
        self.radii[curr_frame] = r


        # Update start and end frames
        if self.start is None:
            self.start = curr_frame
            self.end = curr_frame
        elif curr_frame < self.start:
            self.start = curr_frame
        elif curr_frame > self.end:
            self.end = curr_frame

    def get_by_frame(self, curr_frame: int) -> tuple[NDArray, float]:
        """
        Retrieve ball coordinates and radius at a specific frame.

        Args:
            curr_frame (int): Frame index to retrieve.

        Returns:
            tuple: (coord, radius) for that frame.
        """

        if curr_frame > self.n_frames:
            raise Exception("Trying to access an out of bound frame")
        return self.image_points[curr_frame], self.radii[curr_frame]

    def get_coords(self, start: int | None = None, end: int | None = None) -> NDArray:
        """
        Get coordinates in a specified frame range or full trajectory.

        Args:
            start (int | None): Start frame (default = self.start).
            end (int | None): End frame (default = self.end).

        Returns:
            NDArray: Array of coordinates in the specified range.
        """

        return self.image_points[start or self.start : end or self.end]

    def get_radii(self, start: int | None = None, end: int | None = None) -> NDArray:
        """
        Get radii in a specified frame range or full trajectory.

        Args:
            start (int | None): Start frame (default = self.start).
            end (int | None): End frame (default = self.end).

        Returns:
            NDArray: Array of radii in the specified range.
        """

        return self.radii[start or self.start: end or self.end]

    @staticmethod
    def interpolate_array(arr, window=8, inverse_fit=False):
        """
        Interpolate or extrapolate missing (None) values in an array using a linear fit
        over a sliding window.

        Args:
           arr (list): List of numeric values or None.
           window (int): Number of past points to consider for extrapolation.
           inverse_fit (bool): If True, fits 1/y = a*x + b instead of y = a*x + b.

        Returns:
           list: Array with interpolated/extrapolated integer values.
        """

        n = len(arr)
        arr_interp = arr.copy()

        # Find the first known value
        first_known = next((i for i, v in enumerate(arr) if v is not None), None)
        if first_known is None:
            return arr_interp

        # Interpolate or extrapolate after the first known value
        for i in range(first_known + 1, n):
            if arr_interp[i] is None:
                # Collect last 'window' known points
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

        # Convert all known values to int
        for i in range(first_known, n):
            if arr_interp[i] is not None:
                arr_interp[i] = int(arr_interp[i])

        return arr_interp

    def interpolate_centers_2d(self, window=8):
        """
        Interpolates missing 2D points (x, y) together using linear fit over a sliding window.
        Keeps initial None points untouched and converts all interpolated points to int.

        Args:
            window (int): Number of the previous known points to consider for linear fitting.
        """

        n = len(self.image_points)
        interp_points = self.image_points.copy()  # Copy to avoid modifying original immediately

        # Find index of the first known point where both x and y are not None
        first_known = next((i for i, p in enumerate(interp_points) if p is not None and all(v is not None for v in p)),
                           None)
        if first_known is None:
            return  # All points are None, nothing to interpolate

        # Iterate over all frames after the first known point
        for i in range(first_known + 1, n):
            # Check if the current point is missing (None or contains None)
            if interp_points[i] is None or np.any(interp_points[i] == None):
                # Collect the last 'window' known points before current frame
                known_indices = [j for j in range(max(first_known, i - window), i) if interp_points[j] is not None]

                if len(known_indices) >= 2:
                    # Perform linear fit using last known points
                    t_known = np.array(known_indices)
                    x_known = np.array([interp_points[j][0] for j in known_indices], dtype=float)
                    y_known = np.array([interp_points[j][1] for j in known_indices], dtype=float)

                    # Linear regression: x(t) = a*t + b and y(t) = c*t + d
                    A = np.vstack([t_known, np.ones(len(t_known))]).T
                    x_coeff = np.linalg.lstsq(A, x_known, rcond=None)[0]
                    y_coeff = np.linalg.lstsq(A, y_known, rcond=None)[0]

                    # Interpolate the missing point
                    interp_points[i] = [int(round(np.dot([i, 1], x_coeff))),
                                        int(round(np.dot([i, 1], y_coeff)))]
                elif known_indices:
                    # If only one known point exists, propagate it forward
                    interp_points[i] = [int(interp_points[known_indices[-1]][0]),
                                        int(interp_points[known_indices[-1]][1])]
                else:
                    # If no previous known points, use the next known point if available
                    next_known = next((interp_points[j] for j in range(i + 1, n) if interp_points[j] is not None), None)
                    if next_known is not None:
                        interp_points[i] = [int(next_known[0]), int(next_known[1])]

        # Convert all known points after first_known to integers
        for i in range(first_known, n):
            if interp_points[i] is not None:
                interp_points[i] = [int(interp_points[i][0]), int(interp_points[i][1])]

        # Update class attribute with interpolated points
        self.image_points = interp_points

    def interpolate_all(self):
        """
        Interpolates both radii and 2D centers for the trajectory:
        - Radii: Uses inverse fit (for perspective scaling effects)
        - Centers: Interpolates x and y coordinates together
        """

        self.radii = self.interpolate_array(self.radii, window=10, inverse_fit=True)
        self.interpolate_centers_2d(window=8)

    def plot_onto(self, image: MatLike) -> None:
        """
        Plots the trajectory on an image as circles and a connecting polyline.

        Args:
            image: The image (OpenCV Mat) to draw the trajectory on.
        """

        to_plot = []
        for curr_frame, curr_pos in enumerate(self.image_points):
            curr_rad = self.radii[curr_frame]

            # Skip points if coordinates are None or NaN
            if curr_pos[0] is None or curr_pos[1] is None:
                continue
            if np.isnan(curr_pos[0]) or np.isnan(curr_pos[1]):
                continue

            cx, cy = map(int, curr_pos)

            # Draw circle if radius is valid
            if curr_rad is not None and not np.isnan(curr_rad):
                cv.circle(image, (cx, cy), int(curr_rad), self.color, 1)

            # Collect points for polyline
            to_plot.append((cx, cy))

        # Draw a polyline connecting all valid points
        if to_plot:
            to_plot = np.array(to_plot, dtype=np.int32).reshape((-1, 1, 2))
            cv.polylines(image, [to_plot], isClosed=False, color=self.color, thickness=2)

class BallTrajectory3d:
    """
       Represents the 3D trajectory of a ball over a sequence of frames.
    """

    def __init__(self, n_frames: int, fps: int | None = None, coords: NDArray | None = None, radius: int | None = None, start: int | None = None, end: int | None = None):
        """
        Initializes a BallTrajectory3d instance.

        Args:
           n_frames: Total number of frames to track.
           fps: Frame rate of the video (optional).
           coords: Predefined array of 3D coordinates (optional).
           radius: Ball radius (optional).
           start: Initial frame where the ball appears (optional).
           end: Last frame where the ball appears (optional).
        """

        self.n_frames = n_frames
        self.start = start
        self.end = end
        self.fps = fps

        # If coordinates are provided, ensure their length matches number of frames
        if coords is not None:
            assert len(coords) == n_frames

        # If no coordinates provided, initialize as a n_frames x 3 array with None values
        self.coords = coords or np.array([[None, None, None]] * n_frames)
        self.radius = radius

        # Assign a random color for visualization purposes
        self.color = (random.randint(0, 255), random.randint(0, 255),random.randint(0, 255))

    def set_by_frame(self, coord: NDArray, curr_frame: int) -> None:
        """
        Sets the 3D coordinate for a specific frame.

        Args:
            coord: 3D coordinate array [x, y, z].
            curr_frame: Frame index to set the coordinate for.

        Raises:
            Exception: If the frame index is out of bounds.
        """

        if curr_frame > self.n_frames:
            raise Exception("Trying to access an out of bounds frame")

        self.coords[curr_frame] = coord

        # Update start and end frames based on current frame
        if self.start is None:
            self.start = curr_frame
            self.end = curr_frame
        elif curr_frame < self.start:
            self.start = curr_frame
        elif curr_frame > self.end:
            self.end = curr_frame

    def get_by_frame(self, curr_frame: int) -> NDArray:
        """
        Returns the 3D coordinate for a specific frame.

        Args:
            curr_frame: Frame index to retrieve.

        Returns:
            3D coordinate array [x, y, z].

        Raises:
            Exception: If the frame index is out of bounds.
        """

        if curr_frame > self.n_frames:
            raise Exception("Trying to access an out of bounds frame")
        return self.coords[curr_frame]

    def get_coords(self, start: int | None = None, end: int | None = None):
        """
        Returns the 3D coordinates for a range of frames.

        Args:
            start: Start frame index (defaults to first non-None frame).
            end: End frame index (defaults to last non-None frame).

        Returns:
            Subarray of 3D coordinates from the start to the end frames.
        """

        start = start or self.start
        end = end or self.end
        return self.coords[start:end]

class View:
    """
    Represents a single camera view of a video, optionally including lane information and a 2D ball trajectory.
    """
    def __init__(self, camera: Camera, video: Video, lane: Lane | None = None, trajectory: BallTrajectory2d | None = None,):
        """
        Initializes a View instance.

        Args:
           camera: Camera object capturing this view.
           video: Video object for this view.
           lane: Optional lane information; defaults to a new Lane object if None.
           trajectory: Optional 2D ball trajectory; defaults to None.
        """
        self.camera = camera
        self.video = video
        self.lane = lane or Lane()
        self.trajectory = trajectory

class Environment:
    """
    A global environment manager for handling videos, cameras, and pipelines.
    Stores global paths, video/camera names, and environment variables.
    Provides static methods to initialize and access views and run processing pipelines.
    """

    CV_VISUALIZATION_NAME = "OpenCV Visualization"
    video_names: list[str] = []
    camera_names: list[str] = []
    paths: dict[str, str] = {}
    env_vars: dict[str, object] = {}
    save_name: str = None
    __initialized: bool = False

    @staticmethod
    def initialize_globals(save_name: str, global_parameters: dict) -> None:
        """
        Initializes global environment variables.

        Args:
            save_name: Name to use when saving processed data.
            global_parameters: Dictionary containing 'video_names', 'camera_names',
                               'paths', 'coords', and 'visualization' flags.
        """

        print(f"Saves as : {save_name}")
        Environment.save_name = save_name
        Environment.video_names = global_parameters.get("video_names", [])
        Environment.camera_names = global_parameters.get("camera_names", [])
        Environment.paths = global_parameters.get("paths", {})
        Environment.env_vars = {}
        Environment.coords = global_parameters.get("coords", {})
        Environment.visualization = global_parameters.get("visualization", False)
        print("Default visualization : ", Environment.visualization)
        Environment.__initialized = True

        # Load the views after initialization
        Environment.__load_views()

    @staticmethod
    def __load_views() -> None:
        """
        Initializes View objects for each camera and video based on global parameters.
        Stores each view in env_vars keyed by camera name.
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
        """
        Executes a series of processing pipeline stages (pipes) according to pipeline_configs.

        Args:
           pipeline_configs: List of dictionaries specifying pipe 'name', 'type', and optional 'params'.
        """

        # Map pipe names to classes
        pipes: dict = {
            "intrinsic": cal.IntrinsicCalibration,
            "video_synchronization": vp.SynchronizeVideo,
            "video_undistortion": vp.UndistortVideo,
            "lane_detection": loc.DetectLane,
            "extrinsic": cal.ExtrinsicCalibration,
            "ball_tracker": loc.TrackBall,
            "ball_localization": loc.LocalizeBall,
            "ball_rotation": loc.SpinBall
        }

        # Instantiate pipe objects with the current save_name
        for key in pipes.keys():
            pipes.update({key: pipes.get(key)(Environment.save_name)})

        # Execute or load each pipeline stage as specified
        for pipe_conf in pipeline_configs:
            pipe = pipes[pipe_conf["name"]]
            params: dict = pipe_conf.get("params", None)
            proc_type: str = pipe_conf["type"]
            print(f"\n>>>>> {pipe.__class__.__name__} | {proc_type} <<<<<\n       params : {params}\n")
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
        Retrieves an object stored in the environment.

        Args:
            name: Name of the object to retrieve.

        Returns:
            The object stored under 'name' in env_vars.

        Raises:
            Exception: If the environment is not initialized or the name does not exist.
        """

        if not Environment.__initialized:
            raise Exception("Environment has not been initialized")
        if name in Environment.env_vars:
            return Environment.env_vars.get(name)
        raise Exception(f'Trying to access "{name}" in Environment while it does not exists')

    @staticmethod
    def set(name: str, obj: object, overwrite: bool = False) -> None:
        """
        Stores an object in the environment.

        Args:
            name: Name under which to store the object.
            obj: Object to store.
            overwrite: Whether to overwrite an existing object with the same name.

        Raises:
            Exception: If environment not initialized or attempting to overwrite without permission.
        """

        if not Environment.__initialized:
            raise Exception("Environment has not been initialized")
        if not overwrite and name in Environment.env_vars:
            raise Exception("Trying to overwrite without permission")
        Environment.env_vars[name] = obj

    @staticmethod
    def get_views() -> list[View]:
        """
        Returns:
            List of all View objects corresponding to the initialized cameras.
        """

        return [Environment.get(name) for name in Environment.camera_names]

class DataManager:
    """
    A simple utility class for saving, loading, and deleting Python objects
    using pickle serialization.
    """

    save_path: str = "resources/pickle/"

    @staticmethod
    def save(obj: object, save_name: str) -> None:
        """
        Saves an object to disk using pickle.

        Args:
            obj: The Python object to save.
            save_name: Name of the file (without extension) to save the object as.
        """

        with open(f"{DataManager.save_path}{save_name}.pkl", "wb") as out:
            pickle.dump(obj, out, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(save_name: str) -> None:
        """
        Loads an object from a pickle file.

        Args:
            save_name: Name of the file (without extension) to load the object from.

        Returns:
            The deserialized Python object.

        Raises:
            Exception: If the file does not exist.
        """

        try:
            with open(f"{DataManager.save_path}{save_name}.pkl", "rb") as inp:
                obj = pickle.load(inp)
                return obj
        except FileNotFoundError:
            raise Exception(f"Failed to find load : {save_name}")

    @staticmethod
    def delete(save_name: str) -> None:
        """
        Deletes a saved pickle file.

        Args:
            save_name: Name of the file (without extension) to delete.
        """

        print(f"Deleting {save_name}")
        os.remove(f"{DataManager.save_path}{save_name}.pkl")
        print(f"> {save_name} deleted")