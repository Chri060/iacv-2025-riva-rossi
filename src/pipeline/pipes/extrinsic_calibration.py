from typing import cast

import cv2 as cv
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from matplotlib import pyplot as plt

import pipeline.plot_utils as plot_utils
from pipeline.environment import DataManager, Environment
from pipeline.pipe import Pipe


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

        # Save path
        try:
            save_path = params["save_path"]
        except Exception as _:
            raise Exception("Missing required parameter : save_path")

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
            _, rotation_vector, translation_vector = cv.solvePnP(world_points, image_points, intrinsic, None,
                                                                 flags=cv.SOLVEPNP_ITERATIVE)

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
            ext_calibration_results.update({view.camera.name: {"extrinsic": extrinsic, "position": camera_position,
                                                               "rotation": camera_orientation}})

        # Visualization
        if visualization:
            self.visualize()

        # Save the results
        DataManager.save(ext_calibration_results, self.save_name)

        ax = plot_utils.get_3d_plot("Camera placement : 3D Visualization")

        # Draw bowling lane
        plot_utils.bowling_lane(ax, np.array(Environment.coords["world_lane"]))

        # Draw world reference axes
        plot_utils.reference_frame(
            ax,
            [0, 0, 0],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            label="World reference frame",
            lcolor="cyan"
        )

        # Draw cameras in 3D space
        for view in Environment.get_views():
            plot_utils.camera(ax, view.camera)

        # Define views: (elev, azim)
        views = {
            "front": (20, 0),
            "back": (20, 180),
            "top": (90, -90),
            "side": (0, 90)
        }

        # Save each view without showing
        for name, (elev, azim) in views.items():
            ax.view_init(elev=elev, azim=azim)
            plt.savefig(f"{save_path}/{Environment.save_name}_{name}_{Environment.video_name.removesuffix(".mp4")}.png")

        plt.close(ax.figure)

        input("\033[92mPress Enter to continue...\033[0m")

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
        ext_calibration_results = cast(dict, DataManager.load(self.save_name))

        # Restore camera parameters
        for view in Environment.get_views():
            res = ext_calibration_results[view.camera.name]
            view.camera.extrinsic = res["extrinsic"]
            view.camera.position = res["position"]
            view.camera.rotation = res["rotation"]

        # Visualization
        if visualization:
            self.visualize()

        input("\033[92mPress Enter to continue...\033[0m")

    def plotly_page(self, _):
        """
        Creates a Plotly 3D visualization page:
        - Bowling lane mesh
        - Camera cones showing position and orientation
        - Pit highlighted in red
        """

        ext_calibration_results = cast(dict, DataManager.load(self.save_name))

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
                go.Cone(x=pos1[0], y=pos1[1], z=pos1[2], u=[rot1[0]], v=[rot1[1]], w=[rot1[2]], sizemode="absolute",
                        sizeref=1, showscale=False, name=views[0].camera.name, colorscale=[[0, "red"], [1, "red"]],
                        cmin=0, cmax=1),
                go.Cone(x=pos2[0], y=pos2[1], z=pos2[2], u=[rot2[0]], v=[rot2[1]], w=[rot2[2]], sizemode="absolute",
                        sizeref=1, showscale=False, name=views[1].camera.name, colorscale=[[0, "blue"], [1, "blue"]],
                        cmin=0, cmax=1),
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
        plot_utils.reference_frame(ax, [0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], label="World reference frame",
                                   lcolor="cyan")

        # Draw cameras in 3D space
        for view in Environment.get_views():
            plot_utils.camera(ax, view.camera)

        # Show plot
        plot_utils.show()
