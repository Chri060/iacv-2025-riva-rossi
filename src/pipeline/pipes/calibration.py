import glob
import os
import random

import cv2 as cv
import numpy as np
import plotly.graph_objects as go
from cv2.typing import MatLike
from dash import dcc, html

import pipeline.plot_utils as plot_utils
from pipeline.environment import DataManager, Environment
from pipeline.pipe import Pipe


class Intrinsic_Calibration(Pipe):
    def __process_params(self, params: dict):
        try:
            images_path = params["images_path"]
        except Exception as _:
            raise Exception("Missing required parameter : images_path")
        try:
            visualization = params.get("visualization", Environment.visualization)
        except AttributeError as _:
            visualization = Environment.visualization
        checkerboard_sizes = params.get("checkerboard_sizes", [[9, 6], [9, 6]])
        return images_path, visualization, checkerboard_sizes

    def execute(self, params: dict):
        images_path, visualization, checkerboard_sizes = self.__process_params(params)

        calibration_results = []
        for i, camera_name in enumerate(Environment.camera_names):
            checkerboard_size = checkerboard_sizes[i]
            # Termination criteria for corner sub-pixel refinement
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            # Prepare object points based on the checkerboard grid
            world_points = np.zeros(
                (checkerboard_size[0] * checkerboard_size[1], 3), np.float32
            )
            world_points[:, :2] = np.mgrid[
                0 : checkerboard_size[0], 0 : checkerboard_size[1]
            ].T.reshape(-1, 2)

            # Lists to store object points and image points from all images
            object_points = []  # 3D points in real-world space
            image_points = []  # 2D points in image plane
            images = glob.glob(os.path.join(images_path, camera_name, "*.jpg"))

            print(f"Calibrating {camera_name} ...")

            if images:
                for input_image in images:
                    img = cv.imread(input_image)
                    if img is None:
                        print(f"Error reading image: {input_image}")
                        continue

                    refined_corners = self.__find_checkerboard(
                        img, checkerboard_size, criteria, visualization
                    )

                    if refined_corners is not None:
                        object_points.append(world_points)
                        image_points.append(refined_corners)

                        img_shape = img.shape[:2]

            videos = glob.glob(os.path.join(images_path, camera_name, "*.mp4"))
            if videos :
                for video in videos:
                    capture = cv.VideoCapture(video)
                    total_frames = int(
                        capture.get(cv.CAP_PROP_FRAME_COUNT)
                    )  # Get total number of frames

                    if total_frames < 60:
                        print(f"Warning: {video} has less than 60 frames.")

                    frame_indices = sorted(
                        random.sample(range(total_frames), min(60, total_frames))
                    )  # Select 60 unique random frames

                    for idx in frame_indices:
                        capture.set(
                            cv.CAP_PROP_POS_FRAMES, idx
                        )  # Jump to the selected frame
                        ret, frame = capture.read()

                        if not ret:
                            print(f"Unable to read frame {idx} from {video}")
                            continue

                        refined_corners = self.__find_checkerboard(
                            frame, checkerboard_size, criteria, visualization
                        )

                        if refined_corners is not None:
                            object_points.append(world_points)
                            image_points.append(refined_corners)
                            img_shape = frame.shape[
                                :2
                            ]  # Ensure img_shape is updated correctly

                    capture.release()

            print(f"Checkerboard matched : {len(image_points)}")

            # Proceed with calibration if at least one checkerboard was detected
            if len(object_points) > 0:
                ret, mtx, dist, _, _ = cv.calibrateCamera(
                    object_points, image_points, img_shape, None, None
                )
                view = Environment.get(camera_name)
                view.camera.intrinsic = mtx
                view.camera.distortion = dist
                calibration_results.append(
                    {"camera_name": camera_name, "intrinsic": mtx, "distortion": dist}
                )
            else:
                raise Exception("No valid checkerboard detections. Calibration failed.")

        # Save the calibration_results at the end of the calibration process for every view
        DataManager.save(calibration_results, self.save_name)

    def __find_checkerboard(
        self,
        img: MatLike,
        checkerboard_size: tuple[int, int],
        criteria: int,
        visualization: bool,
    ):
        # Convert image to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Apply histogram equalization for better contrast
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Define flags for the chessboard detection
        flags = (
            cv.CALIB_CB_ADAPTIVE_THRESH
            + cv.CALIB_CB_FAST_CHECK
            + cv.CALIB_CB_NORMALIZE_IMAGE
        )

        # Detect the checkerboard
        ret, corners = cv.findChessboardCorners(gray, checkerboard_size, flags)

        if ret:
            # Store object points
            # Refine corner locations for better accuracy
            refined_corners = cv.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            # Draw detected corners on the image for visualization
            if visualization:
                cv.drawChessboardCorners(img, checkerboard_size, refined_corners, ret)
                to_plot = cv.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)
                cv.imshow(Environment.CV_VISUALIZATION_NAME, to_plot)
                cv.waitKey(1)

            return refined_corners
        return None

    def load(self, params: dict):
        cal_results = DataManager.load(self.save_name)
        for res in cal_results:
            view = Environment.get(res["camera_name"])
            view.camera.intrinsic = res["intrinsic"]
            view.camera.distortion = res["distortion"]

    # No plotly visualization for intrinsic calibration
    def plotly_page(self, params: dict):
        return None


class Extrinsic_Calibration(Pipe):
    def execute(self, params: dict):
        try:
            visualization = params.get("visualization", Environment.visualization)
        except AttributeError as _:
            visualization = Environment.visualization

        world_points = np.array(Environment.coords["world_lane"])

        ext_calibration_results = {}
        for view in Environment.get_views():
            image_points = view.lane.corners
            intrinsic = np.array(view.camera.intrinsic)
            # Find rotation and translation vectors with PnP without distorsion
            _, rotation_vector, translation_vector = cv.solvePnP(
                world_points, image_points, intrinsic, None, flags=cv.SOLVEPNP_ITERATIVE
            )

            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv.Rodrigues(rotation_vector)

            # Compute camera position in world coordinates
            camera_position = -rotation_matrix.T @ translation_vector

            # Obtaining camera orientation
            camera_orientation = rotation_matrix.T

            extrinsic = np.hstack((rotation_matrix, translation_vector))

            # Updating Environment Camera
            view.camera.extrinsic = extrinsic
            view.camera.position = camera_position
            view.camera.rotation = camera_orientation
            ext_calibration_results.update(
                {
                    view.camera.name: {
                        "extrinsic": extrinsic,
                        "position": camera_position,
                        "rotation": camera_orientation,
                    }
                }
            )

        if visualization:
            self.visualize()

        DataManager.save(ext_calibration_results, self.save_name)

    def load(self, params: dict):
        try:
            visualization = params.get("visualization", Environment.visualization)
        except AttributeError as _:
            visualization = Environment.visualization

        ext_calibration_results = DataManager.load(self.save_name)

        for view in Environment.get_views():
            res = ext_calibration_results[view.camera.name]
            view.camera.extrinsic = res["extrinsic"]
            view.camera.position = res["position"]
            view.camera.rotation = res["rotation"]

        if visualization:
            self.visualize()

    def visualize(self):
        ax = plot_utils.get_3d_plot("Camera placement : 3D Visualization")
        plot_utils.bowling_lane(ax, np.array(Environment.coords["world_lane"]))
        plot_utils.reference_frame(
            ax,
            [0, 0, 0],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            label="World reference frame",
            lcolor="cyan",
        )
        for view in Environment.get_views():
            plot_utils.camera(ax, view.camera)
        plot_utils.show()

    def plotly_page(self, params: dict):
        ext_calibration_results = DataManager.load(self.save_name)

        views = Environment.get_views()
        for view in views:
            res = ext_calibration_results[view.camera.name]
            view.camera.extrinsic = res["extrinsic"]
            view.camera.position = res["position"]
            view.camera.rotation = res["rotation"]

        lane_pos = np.array(Environment.coords["world_lane"])
        x = lane_pos[:, 0]
        y = lane_pos[:, 1]
        z = lane_pos[:, 2]

        # camera centers
        pos1 = views[0].camera.position
        pos2 = views[1].camera.position
        # camera orientation (only the Z camera axis)
        rot1 = views[0].camera.rotation[:, 2]
        rot2 = views[1].camera.rotation[:, 2]

        lane = go.Figure(
            data=[
                go.Mesh3d(
                    x=x, y=y, z=z, color="lightblue", opacity=0.8, name="Bowling Lane"
                ),  # lane
                go.Cone(
                    x=pos1[0],
                    y=pos1[1],
                    z=pos1[2],
                    u=[rot1[0]],
                    v=[rot1[1]],
                    w=[rot1[2]],
                    sizemode="absolute",
                    sizeref=1,
                    showscale=False,
                    name=views[0].camera.name,  # camera 1 direction
                    colorscale=[[0, "red"], [1, "red"]],  # Red color
                    cmin=0,
                    cmax=1,
                ),  # Trick to force a single color
                go.Cone(
                    x=pos2[0],
                    y=pos2[1],
                    z=pos2[2],
                    u=[rot2[0]],
                    v=[rot2[1]],
                    w=[rot2[2]],
                    sizemode="absolute",
                    sizeref=1,
                    showscale=False,
                    name=views[1].camera.name,  # camera 2 direction
                    colorscale=[[0, "blue"], [1, "blue"]],  # Red color
                    cmin=0,
                    cmax=1,
                ),  # Trick to force a single color
                go.Scatter3d(
                    x=x[1:3],
                    y=y[1:3],
                    z=z[1:3],
                    mode="lines",
                    name="Pit",
                    line=dict(width=5, color="red"),
                ),  # end of the bowling lane
            ],
        )

        lane.update_scenes(aspectmode="data")

        lane.update_layout(title="Bowling Lane")

        graph = dcc.Graph(figure=lane, style={"width": "100%", "height": "95vh"})

        page = html.Div(children=graph)

        return {self.__class__.__name__: page}
