import glob, os, random

import cv2 as cv
import numpy as np
import plotly.graph_objects as go
from cv2.typing import MatLike
from dash import dcc, html
from typing import Tuple, cast

import pipeline.plot_utils as plot_utils
from pipeline.environment import DataManager, Environment
from pipeline.pipe import Pipe

class IntrinsicCalibration(Pipe):
    """
    This class performs intrinsic camera calibration using images or video frames
    of a checkerboard pattern. The calibration estimates the intrinsic parameters
    (camera matrix) and distortion coefficients for each camera defined in Environment.
    """

    @staticmethod
    def __process_params(params: dict):
        """
        Process and validate input parameters.
        Required:
            - images_path: path to directory containing calibration images/videos
        Optional:
            - visualization: enable OpenCV visualization (default: Environment.visualization)
            - checkerboard_sizes: list of checkerboard sizes for each camera
        """

        # Image path
        try:
            images_path = params["images_path"]
        except Exception as _:
            raise Exception("Missing required parameter : images_path")

        # Visualization flag
        try:
            visualization = params.get("visualization", Environment.visualization)
        except AttributeError as _:
            visualization = Environment.visualization

        # Checkerboard size per camera (default: 9x6 for both cameras)
        checkerboard_sizes = params.get("checkerboard_sizes", [[9, 6], [9, 6]])

        return images_path, visualization, checkerboard_sizes

    def execute(self, params: dict):
        """
        Main execution method: loops through cameras and performs intrinsic calibration.
        """

        images_path, visualization, checkerboard_sizes = self.__process_params(params)

        calibration_results = []
        for i, camera_name in enumerate(Environment.camera_names):
            checkerboard_size = cast(Tuple[int, int], tuple(checkerboard_sizes[i]))

            # Stopping criteria for corner refinement (sub-pixel accuracy)
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            # Generate "world points" based on checkerboard grid (Z=0 plane)
            world_points = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
            world_points[:, :2] = np.mgrid[0 : checkerboard_size[0], 0 : checkerboard_size[1]].T.reshape(-1, 2)

            # Storage for calibration
            object_points: list[np.ndarray] = []
            image_points: list[np.ndarray] = []
            img_shape: tuple[int, int] | None = None

            # Find all images (*.jpg) for this camera
            images = glob.glob(os.path.join(images_path, camera_name, "*.jpg"))
            print(f"Calibrating {camera_name} ...")

            # Process calibration images
            if images:
                for input_image in images:
                    img = cv.imread(input_image)

                    if img is None:
                        print(f"Error reading image: {input_image}")
                        continue

                    refined_corners = self.__find_checkerboard(img, checkerboard_size, criteria, visualization)

                    if refined_corners is not None:
                        object_points.append(world_points)
                        image_points.append(refined_corners)

                        img_shape = img.shape[:2]

            # Process calibration videos
            videos = glob.glob(os.path.join(images_path, camera_name, "*.mp4"))

            if videos :
                for video in videos:
                    capture = cv.VideoCapture(video)
                    total_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

                    if total_frames < 60:
                        print(f"Warning: {video} has less than 60 frames.")

                    frame_indices = sorted(random.sample(range(total_frames), min(60, total_frames)))

                    for idx in frame_indices:
                        capture.set(cv.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = capture.read()

                        if not ret:
                            print(f"Unable to read frame {idx} from {video}")
                            continue

                        refined_corners = self.__find_checkerboard(frame, checkerboard_size, criteria, visualization)

                        if refined_corners is not None:
                            object_points.append(world_points)
                            image_points.append(refined_corners)
                            img_shape = frame.shape[:2]

                    capture.release()

            print(f"Checkerboard matched : {len(image_points)}")

            # Run calibration
            if len(object_points) > 0 and img_shape is not None:
                ret, mtx, dist, _, _ = cv.calibrateCamera(object_points, image_points, img_shape, None, None)

                # Save results in Environment for later use
                view = Environment.get(camera_name)
                view.camera.intrinsic = mtx
                view.camera.distortion = dist
                calibration_results.append({"camera_name": camera_name, "intrinsic": mtx, "distortion": dist})
            else:
                raise Exception("No valid checkerboard detections. Calibration failed.")

        # Save results
        DataManager.save(calibration_results, self.save_name)

    @staticmethod
    def __find_checkerboard(img: MatLike, checkerboard_size: Tuple[int, int], criteria: Tuple[int, int, float], visualization: bool):
        """
        Detects a checkerboard in an image, refines corner detection,
        and optionally visualizes the result.
        """

        # Convert to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Improve contrast using CLAHE
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Detection flags for robustness
        flags = (cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

        # Detect checkerboard corners
        ret, corners = cv.findChessboardCorners(gray, checkerboard_size, flags)

        if ret:
            # Refine corners for sub-pixel accuracy
            refined_corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Visualization
            if visualization:
                cv.drawChessboardCorners(img, checkerboard_size, refined_corners, ret)
                to_plot = cv.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)
                cv.imshow(Environment.CV_VISUALIZATION_NAME, to_plot)
                cv.waitKey(1)

            return refined_corners

        return None

    def load(self):
        """
        Load calibration results from storage and apply them to Environment views.
        """
        cal_results = DataManager.load(self.save_name)
        for res in cal_results:
            view = Environment.get(res["camera_name"])
            view.camera.intrinsic = res["intrinsic"]
            view.camera.distortion = res["distortion"]

class ExtrinsicCalibration(Pipe):
    """
    This class performs extrinsic camera calibration.
    It estimates each camera's position and orientation in the world coordinate system
    using 3D-2D point correspondences and PnP (Perspective-n-Point).
    """

    def execute(self, params: dict):
        """
        Main execution method:
        - Computes extrinsic parameters (rotation, translation) for all views
        - Updates Environment cameras
        - Optionally visualizes the results
        """

        # Visualization flag
        try:
            visualization = params.get("visualization", Environment.visualization)
        except AttributeError as _:
            visualization = Environment.visualization

        # Load known 3D world coordinates of the bowling lane corners
        world_points = np.array(Environment.coords["world_lane"])

        ext_calibration_results = {}

        # Loop over each camera view
        for view in Environment.get_views():
            # 2D image corners manually labeled
            image_points = view.lane.corners

            # Camera intrinsic matrix (from the intrinsic calibration step)
            intrinsic = np.array(view.camera.intrinsic)

            # Estimate rotation and translation using PnP
            _, rotation_vector, translation_vector = cv.solvePnP(world_points, image_points, intrinsic, None, flags=cv.SOLVEPNP_ITERATIVE)

            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv.Rodrigues(rotation_vector)

            # Compute camera position in world coordinates
            camera_position = -rotation_matrix.T @ translation_vector

            # Set the height of the cameras to correct the planar scene inaccuracies
            camera_position[2] = 1.63
            
            # Camera orientation (world-to-camera rotation)
            camera_orientation = rotation_matrix.T

            # Build full extrinsic matrix
            extrinsic = np.hstack((rotation_matrix, translation_vector))

            # Update Environment camera attributes
            view.camera.extrinsic = extrinsic
            view.camera.position = camera_position
            view.camera.rotation = camera_orientation

            # Save results
            ext_calibration_results.update({view.camera.name: {"extrinsic": extrinsic, "position": camera_position, "rotation": camera_orientation}})

        # Visualization
        if visualization:
            self.visualize()

        # Save the results
        DataManager.save(ext_calibration_results, self.save_name)

    def load(self, params: dict):
        """
        Loads extrinsic calibration results from storage,
        applies them to Environment cameras, and optionally visualizes.
        """

        try:
            visualization = params.get("visualization", Environment.visualization)
        except AttributeError as _:
            visualization = Environment.visualization

        # Load saved calibration
        ext_calibration_results = DataManager.load(self.save_name)

        # Restore camera parameters
        for view in Environment.get_views():
            res = ext_calibration_results[view.camera.name]
            view.camera.extrinsic = res["extrinsic"]
            view.camera.position = res["position"]
            view.camera.rotation = res["rotation"]

        # Visualization
        if visualization:
            self.visualize()

    @staticmethod
    def visualize():
        """
        Generates a 3D Matplotlib plot showing:
        - Bowling lane
        - World reference frame
        - All calibrated cameras (position and orientation)
        """

        ax = plot_utils.get_3d_plot("Camera placement : 3D Visualization")

        # Draw bowling lane
        plot_utils.bowling_lane(ax, np.array(Environment.coords["world_lane"]))

        # Draw world reference axes
        plot_utils.reference_frame(ax, [0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], label="World reference frame", lcolor="cyan")

        # Draw cameras in 3D space
        for view in Environment.get_views():
            plot_utils.camera(ax, view.camera)

        # Show plot
        plot_utils.show()

    def plotly_page(self):
        """
        Creates a Plotly 3D visualization page:
        - Bowling lane mesh
        - Camera cones showing position and orientation
        - Pit highlighted in red
        """

        ext_calibration_results = DataManager.load(self.save_name)

        # Update Environment with loaded calibration
        views = Environment.get_views()
        for view in views:
            res = ext_calibration_results[view.camera.name]
            view.camera.extrinsic = res["extrinsic"]
            view.camera.position = res["position"]
            view.camera.rotation = res["rotation"]

        # Bowling lane points
        lane_pos = np.array(Environment.coords["world_lane"])
        x = lane_pos[:, 0]
        y = lane_pos[:, 1]
        z = lane_pos[:, 2]

        # Camera centers
        pos1 = views[0].camera.position
        pos2 = views[1].camera.position

        # Camera orientations (Z-axis as the viewing direction)
        rot1 = views[0].camera.rotation[:, 2]
        rot2 = views[1].camera.rotation[:, 2]

        # Build the 3D scene
        lane = go.Figure(
            data=[
                go.Mesh3d(x=x, y=y, z=z, color="lightblue", opacity=0.8, name="Bowling Lane"),
                go.Cone(x=pos1[0], y=pos1[1], z=pos1[2], u=[rot1[0]], v=[rot1[1]], w=[rot1[2]], sizemode="absolute", sizeref=1, showscale=False, name=views[0].camera.name, colorscale=[[0, "red"], [1, "red"]], cmin=0, cmax=1),
                go.Cone(x=pos2[0], y=pos2[1], z=pos2[2], u=[rot2[0]], v=[rot2[1]], w=[rot2[2]], sizemode="absolute", sizeref=1, showscale=False, name=views[1].camera.name, colorscale=[[0, "blue"], [1, "blue"]], cmin=0, cmax=1),
                go.Scatter3d(x=x[1:3], y=y[1:3], z=z[1:3], mode="lines", name="Pit", line=dict(width=5, color="red"))
            ]
        )

        # Make axes proportional
        lane.update_scenes(aspectmode="data")

        # Title
        lane.update_layout(title="Bowling Lane")

        # Wrap into a dash component
        graph = dcc.Graph(figure=lane, style={"width": "100%", "height": "95vh"})
        page = html.Div(children=graph)

        return {self.__class__.__name__: page}