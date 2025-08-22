from typing import cast

import cv2 as cv
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from scipy.interpolate import make_smoothing_spline
import matplotlib.pyplot as plt
import pipeline.plot_utils as plot_utils
from pipeline.environment import BallTrajectory3d, DataManager, Environment
from pipeline.pipe import Pipe


class LocalizeBall(Pipe):
    """
    Class to localize a bowling ball in 3D space using stereo camera views.
    """

    def execute(self, params: dict):
        """
        Main execution method that computes the 3D trajectory of the ball.
        """

        save_path = params["save_path"]
        visualization = params.get("visualization", False)
        views = Environment.get_views()

        # Compute the projection matrices for both cameras
        int1 = views[0].camera.intrinsic
        int2 = views[1].camera.intrinsic
        ext1 = views[0].camera.extrinsic
        ext2 = views[1].camera.extrinsic
        proj1 = int1 @ ext1
        proj2 = int2 @ ext2

        # Get the tracked 2D image points for each camera
        points_cam1 = views[0].trajectory.image_points
        points_cam2 = views[1].trajectory.image_points

        # Find the first frame where both cameras have valid points
        start = 0
        while points_cam1[start, 0] is None or points_cam2[start, 0] is None:
            start += 1

        # Prepare points for triangulation: remove None values and transpose
        points1 = points_cam1[start:-1].T.astype(np.float32)
        points2 = points_cam2[start:-1].T.astype(np.float32)

        # Triangulate the 3D points from the two camera views
        homogeneous_points = cv.triangulatePoints(proj1, proj2, points1, points2)
        points_3d = (homogeneous_points[:3] / homogeneous_points[3]).T

        # Keep only points that are over the bowling lane
        lane_coords = Environment.coords["world_lane"]
        minx, miny, _ = min(lane_coords)
        maxx, maxy, _ = max(lane_coords)
        start_over = None
        end_over = None
        for frame, (x, y, _) in enumerate(points_3d):
            if start_over is None and minx <= x <= maxx and miny <= y <= maxy:
                start_over = frame
            elif start_over is not None and end_over is None and (x <= minx or x >= maxx or y <= miny or y >= maxy):
                end_over = frame

        if start_over is None:
            raise Exception("The ball trajectory detected is outside the bowling lane")

        # If the trajectory ends inside the lane, take the end as the last frame
        if end_over is None:
            end_over = len(points_3d)

        points_3d = points_3d[start_over:end_over]

        # Smooth the 3D trajectory using spline interpolation
        t = np.arange(0, len(points_3d), 1)

        spl_x = make_smoothing_spline(t, points_3d[:, 0], lam=100)
        spl_y = make_smoothing_spline(t, points_3d[:, 1], lam=100)
        z = np.full_like(spl_x(t), params.get("ball_radius", 0.1091))  # same shape as spl_x(t)

        # Update points_3d with smoothed coordinates
        points_3d = np.array([spl_x(t), spl_y(t), z]).T.reshape(-1, 3)

        # Store the 3D trajectory in a Ball_Trajectory_3D object
        trajectory_3d = BallTrajectory3d(views[0].trajectory.n_frames)
        start_3d = start + start_over
        end_3d = start + end_over
        for i, frame in enumerate(range(start_3d, end_3d)):
            trajectory_3d.set_by_frame(points_3d[i], frame)

        # Save the trajectory
        Environment.set("3D_trajectory", trajectory_3d)
        DataManager.save(trajectory_3d, self.save_name)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Ball Localization : 3D Visualization")

        # Plot lane and trajectory
        plot_utils.bowling_lane(ax, np.array(Environment.coords["world_lane"]))
        plot_utils.trajectory(ax, Environment.get("3D_trajectory"))

        # Define views: (elev, azim)
        views = {"back": (20, 180), "top": (90, -90), "side": (0, 90), "front": (20, 0)}

        for name, (elev, azim) in views.items():
            ax.view_init(elev=elev, azim=azim)
            plt.savefig(f"{save_path}/{Environment.save_name}_{name}_{Environment.video_name.removesuffix(".mp4")}.png")
        if visualization:
            plt.show()
        else:
            plt.close(fig)

        input("\033[92mPress Enter to continue...\033[0m")

    def load(self, params: dict):
        trajectory_3d = DataManager.load(self.save_name)
        Environment.set("3D_trajectory", trajectory_3d)

        # Get the visualization flag from params or default to environment setting
        visualization = params.get("visualization", False)

        if visualization:
            ax = plot_utils.get_3d_plot("Ball Localization : 3D Visualization")
            plot_utils.bowling_lane(ax, np.array(Environment.coords["world_lane"]))
            plot_utils.trajectory(ax, Environment.get("3D_trajectory"))
            plt.show(block=True)

        input("\033[92mPress Enter to continue...\033[0m")

    def plotly_page(self, params: dict) -> dict[str, html.Div]:
        """
        Returns a Dash HTML Div with a 3D plotly visualization of the trajectory and lane.
        Includes animation of the ball along its trajectory.
        """

        # Load the previously saved 3D trajectory
        trajectory_3d = cast(BallTrajectory3d, DataManager.load(self.save_name))
        Environment.set("3D_trajectory", trajectory_3d)

        # Helper function to create a 3D sphere at a given center with a specified radius
        def make_sphere(center, ball_radius, resolution=10):
            u, v = np.mgrid[0: 2 * np.pi: resolution * 2j, 0: np.pi: resolution * 1j]
            return go.Surface(x=ball_radius * np.cos(u) * np.sin(v) + center[0],
                              y=ball_radius * np.sin(u) * np.sin(v) + center[1], z=ball_radius * np.cos(v) + center[2],
                              opacity=0.6)

        # Extract trajectory coordinates
        xyz_t = trajectory_3d.get_coords()
        xt = xyz_t[:, 0]
        yt = xyz_t[:, 1]
        zt = xyz_t[:, 2]

        # Get bowling lane coordinates
        lane_pos = np.array(Environment.coords["world_lane"])
        x = lane_pos[:, 0]
        y = lane_pos[:, 1]
        z = lane_pos[:, 2]

        # Radius of the bowling ball (in meters)
        radius = params.get("radius", 0.1091)

        # Construct the figure
        lane = go.Figure(
            data=[
                go.Mesh3d(x=x, y=y, z=z, color="lightblue", opacity=0.8, name="Bowling Lane"),
                go.Scatter3d(x=x[1:3], y=y[1:3], z=z[1:3], mode="lines", name="Pit", line=dict(width=5, color="red")),
                go.Scatter3d(x=xt, y=yt, z=zt, mode="lines", name="Trajectory", line=dict(width=5, color="green")),
            ],
            layout=go.Layout(
                updatemenus=[
                    {
                        "type": "buttons",
                        "direction": "left",
                        "x": 1,
                        "y": 1,
                        "buttons": [
                            {
                                "label": "Play",
                                "method": "animate",
                                "args": [
                                    None,
                                    {
                                        "frame": {"duration": 5, "redraw": True},
                                        "fromcurrent": True,
                                        "mode": "immediate",
                                    },
                                ],
                            },
                            {
                                "label": "Pause",
                                "method": "animate",
                                "args": [
                                    None,
                                    {
                                        "frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                    },
                                ],
                            },
                        ],
                    }
                ],
                title="Bowling Lane",
            ),
            frames=[go.Frame(data=make_sphere(pos, radius)) for pos in xyz_t],
        )

        lane.update_scenes(aspectmode="data")

        graph = dcc.Graph(figure=lane, style={"width": "100%", "height": "95vh"})

        page = html.Div(children=graph)

        return {self.__class__.__name__: page}
