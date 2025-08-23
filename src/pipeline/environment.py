import pickle, random, re
from typing import Any, cast
import cv2 as cv
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray


class Camera:
    """
    Class to store camera parameters and metadata.

    A Camera object holds intrinsic, extrinsic, and distortion parameters,
    as well as the 3D position and rotation of the camera in space.

    Attributes:
        name (str): Name or identifier for the camera.
        intrinsic (NDArray | None): Intrinsic camera matrix (3x3) if available.
        distortion (NDArray | None): Distortion coefficients if available.
        extrinsic (NDArray | None): Extrinsic matrix (4x4) defining camera pose if available.
        position (NDArray | None): 3D position of the camera in world coordinates.
        rotation (NDArray | None): 3x3 rotation matrix representing camera orientation.
    """

    def __init__(self, name: str):
        """
        Initialize a Camera object with a given name.

        Args:
            name (str): The name or identifier for the camera.
        """

        self.name = name
        self.intrinsic: NDArray | None = None
        self.distortion: NDArray | None = None
        self.extrinsic: NDArray | None = None
        self.position: NDArray | None = None
        self.rotation: NDArray | None = None


class Video:
    """
    Class to handle video loading and basic property extraction.

    This class provides an interface to load a video using OpenCV and
    extract essential properties such as frame rate, duration, and resolution.

    Attributes:
        capture (cv.VideoCapture): OpenCV VideoCapture object for reading video frames.
        path (str): File path to the video.
    """

    def __init__(self, path: str):
        """
        Initialize a Video object by opening the specified video file.

        Args:
            path (str): The file path to the video.
        """

        self.capture = cv.VideoCapture(path)
        self.path = path

    def get_video_properties(self) -> tuple[float, float | int, tuple[int, int]]:
        """
        Retrieve basic properties of the video.

        Calculates frames per second (fps), total duration in seconds, and
        the resolution (width x height) of the video.

        Returns:
            tuple: A tuple containing:
                - fps (float): Frames per second of the video.
                - duration (float | int): Total duration in seconds.
                - resolution (tuple[int, int]): Width and height of the video in pixels.
        """

        # Get frames per second
        fps = self.capture.get(cv.CAP_PROP_FPS)

        # Get the total number of frames
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

    A Lane object stores geometric information about a lane, such as its
    corner points. The corners can be represented in 2D image coordinates
    or 3D world coordinates.

    Attributes:
        corners (NDArray | None): Array of corner points defining the lane.
            Can be 2D (image coordinates) or 3D (world coordinates). Defaults to None.
    """

    def __init__(self, corners: NDArray | None = None):
        """
        Initialize a Lane object with optional corner points.

        Args:
            corners (NDArray | None, optional): Array of points defining the lane.
                Defaults to None.
        """

        self.corners: NDArray | None = corners


class BallTrajectory2d:
    """
    Class to represent the 2D trajectory of a ball across video frames.

    Stores image coordinates and radii for each frame and provides
    functionality to interpolate missing values, retrieve trajectory data,
    and visualize the trajectory on images.

    Attributes:
        n_frames (int): Total number of frames in the video or trajectory.
        fps (float | None): Frames per second of the video.
        start (int | None): First frame index with valid data.
        end (int | None): Last frame index with valid data.
        image_points (NDArray): Array of shape (n_frames, 2) storing (x, y) coordinates.
        radii (NDArray): Array of shape (n_frames) storing the radius per frame.
        color (tuple[int, int, int]): Random RGB color assigned for visualization.
    """

    def __init__(self, n_frames: int, fps: float | None = None, image_points: NDArray | None = None,
                 radii: NDArray | None = None, start: int | None = None, end: int | None = None):
        """
        Initialize a BallTrajectory2d object.

        Args:
            n_frames (int): Number of frames in the video or trajectory.
            fps (float | None, optional): Frames per second of the video. Defaults to None.
            image_points (NDArray | None, optional): Pre-filled array of (x, y) coordinates.
            radii (NDArray | None, optional): Pre-filled array of ball radii.
            start (int | None, optional): Starting frame of the detected trajectory.
            end (int | None, optional): End frame of the detected trajectory.
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

    def set_by_frame(self, coord: NDArray | None, r: float | None, curr_frame: int) -> None:
        """
        Set the ball's position and radius for a specific frame.

        Updates the start and end frames automatically.

        Args:
            coord (NDArray | None): 2D coordinates (x, y) in the image frame.
            r (float | None): Radius of the ball in the image frame.
            curr_frame (int): Frame index to update.
        """

        if curr_frame > self.n_frames or curr_frame < 0:
            raise Exception("Trying to access an out of bound frame")
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

    def get_by_frame(self, curr_frame: int) -> tuple[
        np.ndarray[tuple[int, ...], np.dtype[Any]], np.ndarray[tuple[int, ...], np.dtype[Any]]]:
        """
        Retrieve the ball coordinates and radius at a specific frame.

        Args:
            curr_frame (int): Frame index to retrieve.

        Returns:
            tuple[NDArray, NDArray]: (coordinates, radius) for that frame.
        """
        if curr_frame > self.n_frames:
            raise Exception("Trying to access an out of bound frame")
        return self.image_points[curr_frame], self.radii[curr_frame]

    def get_coords(self, start: int | None = None, end: int | None = None) -> NDArray:
        """
        Get coordinates in a specified frame range or full trajectory.

        Args:
            start (int | None, optional): Start frame (default = self.start).
            end (int | None, optional): End frame (default = self.end).

        Returns:
            NDArray: Array of coordinates in the specified range.
        """

        return self.image_points[start or self.start: end or self.end]

    def get_radii(self, start: int | None = None, end: int | None = None) -> NDArray:
        """
        Get radii in a specified frame range or full trajectory.

        Args:
            start (int | None, optional): Start frame (default = self.start).
            end (int | None, optional): End frame (default = self.end).

        Returns:
            NDArray: Array of radii in the specified range.
        """

        return self.radii[start or self.start: end or self.end]

    @staticmethod
    def interpolate_array(arr, window=8, inverse_fit=False):
        """
        Interpolate or extrapolate missing values (None) in an array using linear fit.

        Args:
            arr (list): List of numeric values or None.
            window (int, optional): Number of past points to consider. Defaults to 8.
            inverse_fit (bool, optional): If True, fits 1/y = a*x + b. Defaults to False.

        Returns:
            list[int]: Array with interpolated/extrapolated integer values.
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
                        coefficients = np.polyfit(x_known, y_inv, 1)
                        arr_interp[i] = int(round(float(1 / np.polyval(coefficients, i))))
                    else:
                        coefficients = np.polyfit(x_known, y_known, 1)
                        arr_interp[i] = int(round(float(np.polyval(coefficients, i))))
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

        Keeps initial None points untouched and converts interpolated points to int.

        Args:
            window (int, optional): Number of previously known points for linear fitting. Defaults to 8.
        """

        n = len(self.image_points)
        interp_points = self.image_points.copy()

        # Find the first known point
        first_known = next(
            (i for i, p in enumerate(interp_points) if p is not None and all(v is not None for v in p)),
            None
        )
        if first_known is None:
            return  # All points are None

        # Iterate over all frames after the first known point
        for i in range(first_known + 1, n):
            if interp_points[i] is None or any(v is None for v in interp_points[i]):
                # Collect the last 'window' known points before the current index
                known_indices = [
                    j for j in range(max(first_known, i - window), i)
                    if interp_points[j] is not None and all(v is not None for v in interp_points[j])
                ]

                if len(known_indices) >= 2:
                    # Linear regression
                    t_known = np.array(known_indices)
                    x_known = np.array([interp_points[j][0] for j in known_indices], dtype=float)
                    y_known = np.array([interp_points[j][1] for j in known_indices], dtype=float)

                    matrix = np.vstack([t_known, np.ones(len(t_known))]).T
                    x_coef = np.linalg.lstsq(matrix, x_known, rcond=None)[0]
                    y_coef = np.linalg.lstsq(matrix, y_known, rcond=None)[0]

                    # Interpolate / extrapolate
                    interp_points[i] = [
                        int(round(np.dot([i, 1], x_coef))),
                        int(round(np.dot([i, 1], y_coef)))
                    ]
                elif known_indices:
                    # Propagate the last known point
                    last = known_indices[-1]
                    interp_points[i] = [
                        int(interp_points[last][0]),
                        int(interp_points[last][1])
                    ]
                else:
                    # Edge case: no previous known points (should not happen after first_known)
                    next_known = next((interp_points[j] for j in range(i + 1, n)
                                       if
                                       interp_points[j] is not None and all(v is not None for v in interp_points[j])),
                                      None)
                    if next_known is not None:
                        interp_points[i] = [int(next_known[0]), int(next_known[1])]

        # Ensure all points after first_known are integers
        for i in range(first_known, n):
            if interp_points[i] is not None and all(v is not None for v in interp_points[i]):
                interp_points[i] = [int(interp_points[i][0]), int(interp_points[i][1])]

        self.image_points = interp_points

    def interpolate_all(self):
        """
        Interpolates both radii and 2D centers for the trajectory.

        For the radii it uses inverse fit (to account for perspective scaling).
        For the centers it interpolates x and y coordinates together.
        """

        self.radii = self.interpolate_array(self.radii, window=10, inverse_fit=True)
        self.interpolate_centers_2d(window=8)

    def plot_onto(self, image: MatLike) -> None:
        """
        Draws the trajectory on an image.

        Circles are drawn at each frame based on radius, and a polyline
        connects all valid points.

        Args:
            image (MatLike): OpenCV image to draw the trajectory on.
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
    Class to represent the 3D trajectory of a ball over a sequence of frames.

    Stores 3D coordinates for each frame and allows setting, retrieving,
    and visualizing the trajectory. A random color is assigned for visualization.

    Attributes:
        n_frames (int): Total number of frames in the trajectory.
        fps (int | None): Frame rate of the video (optional).
        start (int | None): First frame index with valid coordinates.
        end (int | None): Last frame index with valid coordinates.
        coords (NDArray): Array of shape (n_frames, 3) storing 3D coordinates.
        radius (int | None): Ball radius (optional).
        color (tuple[int, int, int]): Random RGB color for visualization.
    """

    def __init__(self, n_frames: int, fps: int | None = None, coords: NDArray | None = None, radius: int | None = None,
                 start: int | None = None, end: int | None = None):
        """
        Initialize a BallTrajectory3d instance.

        Args:
            n_frames (int): Total number of frames to track.
            fps (int | None, optional): Frame rate of the video. Defaults to None.
            coords (NDArray | None, optional): Predefined array of 3D coordinates. Defaults to None.
            radius (int | None, optional): Ball radius. Defaults to None.
            start (int | None, optional): Initial frame index of the trajectory. Defaults to None.
            end (int | None, optional): Last frame index of the trajectory. Defaults to None.
        """

        self.n_frames = n_frames
        self.start = start
        self.end = end
        self.fps = fps

        # If coordinates are provided, ensure their length matches the number of frames
        if coords is not None:
            assert len(coords) == n_frames

        # If no coordinates provided, initialize as an n_frames x 3 array with None values
        self.coords = coords or np.array([[None, None, None]] * n_frames)
        self.radius = radius

        # Assign a random color for visualization purposes
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def set_by_frame(self, coord: NDArray, curr_frame: int) -> None:
        """
        Set the 3D coordinate for a specific frame.

        Updates the start and end frame indices automatically.

        Args:
            coord (NDArray): 3D coordinate [x, y, z].
            curr_frame (int): Frame index to update.
        """

        if curr_frame > self.n_frames:
            raise Exception("Trying to access an out of bounds frame")

        self.coords[curr_frame] = coord

        # Update start and end frames based on the current frame
        if self.start is None:
            self.start = curr_frame
            self.end = curr_frame
        elif curr_frame < self.start:
            self.start = curr_frame
        elif curr_frame > self.end:
            self.end = curr_frame

    def get_by_frame(self, curr_frame: int) -> NDArray:
        """
        Retrieve the 3D coordinate for a specific frame.

        Args:
            curr_frame (int): Frame index to retrieve.

        Returns:
            NDArray: 3D coordinate [x, y, z] for the specified frame.
        """
        if curr_frame > self.n_frames:
            raise Exception("Trying to access an out of bounds frame")

        return self.coords[curr_frame]

    def get_coords(self, start: int | None = None, end: int | None = None):
        """
        Retrieve 3D coordinates for a range of frames.

        Args:
            start (int | None, optional): Start frame index (default = self.start).
            end (int | None, optional): End frame index (default = self.end).

        Returns:
            NDArray: Array of 3D coordinates from `start` to `end` frames.
        """

        start = start or self.start

        end = end or self.end

        return self.coords[start:end]


class View:
    """
    Class to represent a single camera view of a video, optionally including lane information and a 2D ball trajectory.

    A View object encapsulates all the data associated with one camera perspective,
    including the camera parameters, video, lane geometry, and the 2D trajectory of the ball.

    Attributes:
        camera (Camera): Camera object capturing this view.
        video (Video): Video object associated with this view.
        lane (Lane): Lane geometry information for the view.
        trajectory (BallTrajectory2d | None): 2D trajectory of the ball in this view, if available.
    """

    def __init__(self, camera: Camera, video: Video, lane: Lane | None = None,
                 trajectory: BallTrajectory2d | None = None, ):
        """
        Initialize a View instance.

        Args:
            camera (Camera): Camera capturing this view.
            video (Video): Video associated with this view.
            lane (Lane | None, optional): Lane information. Defaults to a new Lane object if None.
            trajectory (BallTrajectory2d | None, optional): 2D ball trajectory. Defaults to None.
        """

        self.camera = camera
        self.video = video
        self.lane = lane or Lane()
        self.trajectory = trajectory


class Environment:
    """
    Global environment manager for handling videos, cameras, and pipelines.

    Stores global paths, video/camera names, coordinates, and environment variables.
    Provides static methods to initialize views, access stored objects, and run processing pipelines.

    Class Attributes:
        coords (dict): Stores global coordinates or data used by the pipeline.
        visualization (bool): Flag to enable visualization.
        CV_VISUALIZATION_NAME (str): Default name for the OpenCV visualization window.
        video_name (str): Name of the current video being processed.
        camera_names (list[str]): List of camera names in the environment.
        paths (dict[str, str]): Dictionary of global paths (e.g., originals, results).
        env_vars (dict[str, object]): Storage for all objects in the environment keyed by name.
        save_name (str | None): Name used when saving processed data.
        __initialized (bool): Flag indicating whether the environment has been initialized.
    """

    coords = {}
    visualization = None
    CV_VISUALIZATION_NAME = "OpenCV Visualization"
    video_name: str = ""
    camera_names: list[str] = []
    paths: dict[str, str] = {}
    env_vars: dict[str, object] = {}
    save_name: str = None
    __initialized: bool = False

    @staticmethod
    def initialize_globals(save_name: str, global_parameters: dict) -> None:
        """
        Initializes global environment variables and loads camera views.

        Args:
            save_name (str): Name used when saving processed data.
            global_parameters (dict): Dictionary containing:
                - 'video_name': Name of the video.
                - 'camera_names': List of camera names.
                - 'paths': Dictionary of relevant paths.
                - 'coords': Initial coordinates or data.
                - 'visualization': Flag for enabling visualization.
        """

        print(f"Saves as : {save_name}")
        Environment.save_name = save_name
        Environment.video_name = global_parameters.get("video_name", "")
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
        Initializes View objects for each camera and video.
        Each View is stored in env_vars keyed by camera name.
        """

        video_name = Environment.video_name
        camera_names = Environment.camera_names
        originals_path = Environment.paths["originals"]
        for i, camera_name in enumerate(camera_names):
            full_path = f"{originals_path}/{camera_name}/{video_name}"
            video = Video(full_path)
            camera = Camera(camera_name)
            view = View(camera, video)
            Environment.set(camera_name, view)

    @staticmethod
    def start_pipeline(pipeline_configs: dict) -> None:
        """
        Executes a series of pipeline stages according to pipeline_configs.

        Args:
            pipeline_configs (dict): List of dictionaries specifying pipe 'name', 'type',
                                     and optional 'params'. Each dict should contain:
                                     - 'name': The pipeline step name.
                                     - 'type': "execute" or "load".
                                     - 'params': Optional parameters for the pipeline step.
        """

        from pipeline.pipes.intrinsic_calibration import IntrinsicCalibration
        from pipeline.pipes.extrinsic_calibration import ExtrinsicCalibration
        from pipeline.pipes.lane_detection import DetectLane
        from pipeline.pipes.ball_tracking import TrackBall
        from pipeline.pipes.ball_localization import LocalizeBall
        from pipeline.pipes.ball_motion import SpinBall
        from pipeline.pipes.video_synchronization import SynchronizeVideo
        from pipeline.pipes.video_undistortion import UndistortVideo

        # Map pipe names to classes
        pipes: dict = {
            "intrinsic": IntrinsicCalibration,
            "video_synchronization": SynchronizeVideo,
            "video_undistortion": UndistortVideo,
            "lane_detection": DetectLane,
            "extrinsic": ExtrinsicCalibration,
            "ball_tracker": TrackBall,
            "ball_localization": LocalizeBall,
            "ball_rotation": SpinBall
        }

        # Instantiate pipe objects with the current save_name
        for key in pipes.keys():
            pipes.update({key: pipes.get(key)(Environment.save_name)})

        # Execute or load each pipeline stage as specified
        for pipe_conf in pipeline_configs:
            pipe = pipes[pipe_conf["name"]]
            params: dict = pipe_conf.get("params", None)
            proc_type: str = pipe_conf["type"]
            class_name_spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", pipe.__class__.__name__)
            print(f"\n\033[94m>>>>> {class_name_spaced} | {proc_type} <<<<<\033[0m\n       params : {params}\n")
            if proc_type == "execute":
                pipe.execute(params)
            elif proc_type == "load":
                if pipe_conf["name"] == "intrinsic":
                    pipe.load()
                else:
                    pipe.load(params)

        cv.destroyAllWindows()

    @staticmethod
    def get(name: str):
        """
        Retrieve an object stored in the environment.

        Args:
            name (str): Name of the object to retrieve.

        Returns:
            object: The stored object corresponding to the given name.

        Raises:
            Exception: If the environment is not initialized or the object does not exist.
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
            name (str): Name under which to store the object.
            obj (object): Object to store.
            overwrite (bool, optional): Whether to overwrite an existing object with the same name. Defaults to False.

        Raises:
            Exception: If the environment is not initialized or attempting to overwrite without permission.
        """

        if not Environment.__initialized:
            raise Exception("Environment has not been initialized")
        if not overwrite and name in Environment.env_vars:
            raise Exception("Trying to overwrite without permission")
        Environment.env_vars[name] = obj

    @staticmethod
    def get_views() -> list[Any | None]:
        """
        Retrieve all View objects corresponding to the initialized cameras.

        Returns:
            list[object | None]: List of View objects in the same order as camera_names.
        """

        return [Environment.get(name) for name in Environment.camera_names]


class DataManager:
    """
    Utility class for saving, loading, and deleting Python objects using pickle serialization.

    Provides static methods to persist and retrieve Python objects to/from the disk.
    The default storage path is "resources/pickle/".

    Attributes:
        save_path (str): Directory path where pickle files are stored.
    """

    save_path: str = "resources/pickle/"

    @staticmethod
    def save(obj: object, save_name: str, intrinsic: bool = False) -> None:
        """
        Save a Python object to disk using pickle serialization.

        Args:
            obj (object): The Python object to save.
            save_name (str): Name of the file (without extension) to save the object as.
            intrinsic (bool, optional): Whether this file is for intrinsic calibration. Defaults to False.
        """

        if intrinsic:
            with open(f"{DataManager.save_path}/{save_name}.pkl", "wb") as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        else:
            path = f"{DataManager.save_path}/{save_name}_{Environment.video_name.removesuffix('.mp4')}.pkl"
            with open(path, "wb") as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(save_name: str, intrinsic: bool = False) -> object:
        """
        Load a Python object from a pickle file.

        Args:
            save_name (str): Name of the file (without extension) to load the object from.
            intrinsic (bool, optional): Whether the file is for intrinsic calibration. Defaults to False.

        Returns:
            object: The deserialized Python object, or None if the file is not found.
        """

        try:
            if intrinsic:
                with open(f"{DataManager.save_path}/{save_name}.pkl", "rb") as inp:
                    obj = pickle.load(inp)
                    return obj
            else:
                with open(f"{DataManager.save_path}/{save_name}_{Environment.video_name.removesuffix(".mp4")}.pkl",
                          "rb") as inp:
                    obj = pickle.load(inp)
                    return obj
        except FileNotFoundError:
            print(f"Failed to find load : {save_name}")
            return None
