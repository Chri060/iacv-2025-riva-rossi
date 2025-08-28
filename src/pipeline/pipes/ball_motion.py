from typing import cast
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dash import dcc, html
from pipeline import plot_utils
from pipeline.environment import DataManager, Environment, BallTrajectory3d
from pipeline.pipe import Pipe


class SpinBall(Pipe):
    """
    Estimates the spin of a bowling ball along its trajectory using optical flow.

    This pipeline stage:
        1. Loads 2D ball trajectories from video frames.
        2. Detects features on the ball surface and computes optical flow.
        3. Estimates the 3D rotation axis and spin rate (rad/s) per frame.
        4. Saves spin data and optionally visualizes the spin rate and axes.
    """

    def execute(self, params: dict):
        """
        Computes spin rates and rotation axes for the bowling ball.

        Args:
            params (dict): Dictionary of parameters including
                - "graph_save_path" (str): Directory to save plots.
                - "visualization" (bool, optional): Whether to show interactive visualizations.
                - Any other visualization or ball-specific parameters.
        """

        # Load parameters
        graph_save_path = params["graph_save_path"]
        visualization = params.get("visualization", Environment.visualization)

        # Smoothing factor for spin rate estimation
        smoothing_alpha = 0.2

        # Maximum number of features to track per ball
        max_corners = 50

        # Parameters for Lucas-Kanade optical flow
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

        spin_results = {}
        axis_points = {}

        # Iterate over all available camera views
        for view in Environment.get_views():
            cap = view.video.capture
            fps, _, _ = view.video.get_video_properties()
            trajectory = view.trajectory
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)

            spin_rates = []
            old_gray = None
            p0 = None
            frame_idx = 0
            axis_points[view.camera.name] = []

            # Process each video frame
            while True:
                ret, frame = cap.read()
                if not ret or frame_idx >= trajectory.n_frames:
                    break

                frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                vis_frame = frame.copy()
                center, radius = trajectory.get_by_frame(frame_idx)

                if center is not None and radius is not None:
                    cx, cy = map(int, center)
                    radius = int(radius)

                    # Visualize
                    if visualization:
                        cv.circle(vis_frame, (cx, cy), radius, (0, 0, 255), 2)
                        cv.circle(vis_frame, (cx, cy), 3, (0, 0, 255), -1)

                    # Mask ball area for feature detection
                    mask_ball = np.zeros_like(frame_gray)
                    cv.circle(mask_ball, (cx, cy), max(radius - 2, 1), 255, -1)

                    # Detect good features to track inside the ball
                    p_new = cv.goodFeaturesToTrack(frame_gray, mask=mask_ball, maxCorners=max_corners,
                                                   qualityLevel=0.01, minDistance=5)

                    # Optical flow computation if previous features exist
                    if p0 is not None and len(p0) > 0:
                        p1, st, _ = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                        if p1 is not None and st is not None:
                            good_new = p1[st == 1]
                            good_old = p0[st == 1]

                            # Filter features that remain inside the ball region
                            filtered_new, filtered_old = [], []
                            for (new, old) in zip(good_new, good_old):
                                x_new, y_new = new.ravel()
                                if (x_new - cx) ** 2 + (y_new - cy) ** 2 <= (radius - 2) ** 2:
                                    filtered_new.append(new)
                                    filtered_old.append(old)

                            # Compute 3D rotation axis and spin
                            axes_3d = []
                            delta_thetas = []
                            weights = []

                            for (new, old) in zip(filtered_new, filtered_old):
                                # 2D displacement relative to the ball center
                                dx_old, dy_old = old.ravel() - [cx, cy]
                                dx_new, dy_new = new.ravel() - [cx, cy]

                                r_old_len = np.sqrt(dx_old ** 2 + dy_old ** 2)
                                if r_old_len < 0.3 * radius:
                                    continue

                                # Map 2D offsets into 3D local coordinates on the sphere
                                z_old = np.sqrt(max(radius ** 2 - r_old_len ** 2, 0.0))
                                z_new = np.sqrt(max(radius ** 2 - (dx_new ** 2 + dy_new ** 2), 0.0))

                                r_old = np.array([dx_old, dy_old, z_old])
                                r_new = np.array([dx_new, dy_new, z_new])

                                # Rotation axis: cross-product of 3D vectors
                                axis_vec = np.cross(r_old, r_new)
                                if np.linalg.norm(axis_vec) > 1e-6:
                                    axis_vec /= np.linalg.norm(axis_vec)
                                    axes_3d.append(axis_vec)

                                # Rotation angle: arccos of dot product
                                dot = np.dot(r_old, r_new) / (np.linalg.norm(r_old) * np.linalg.norm(r_new))
                                dot = np.clip(dot, -1.0, 1.0)
                                theta = np.arccos(dot)

                                delta_thetas.append(theta)
                                weights.append(r_old_len / radius)

                            # Compute average rotation axis
                            if delta_thetas:
                                avg_axis = np.mean(axes_3d, axis=0) if axes_3d else np.array([0, 0, 1])
                                avg_axis /= np.linalg.norm(avg_axis)
                            else:
                                avg_axis = np.array([0, 0, 1])

                            avg_axis = view.camera.rotation.T @ avg_axis
                            avg_axis /= np.linalg.norm(avg_axis)

                            axis_points[view.camera.name].append(avg_axis)

                            # Compute weighted average spin rate
                            if delta_thetas:
                                med_delta_theta = np.average(delta_thetas, weights=weights)
                                spin_rate = med_delta_theta * fps
                                prev_spin = spin_rates[-1] if spin_rates else 0.0
                                spin_rate = smoothing_alpha * spin_rate + (1 - smoothing_alpha) * prev_spin
                                spin_rates.append(spin_rate)
                            else:
                                spin_rates.append(spin_rates[-1] if spin_rates else 0.0)

                            # Merge new and tracked features
                            if p_new is not None:
                                filtered_new_arr = np.array(filtered_new, dtype=np.float32).reshape(-1, 1, 2)
                                p0 = np.vstack([filtered_new_arr, p_new]) if len(filtered_new) > 0 else p_new
                            else:
                                p0 = np.array(filtered_new, dtype=np.float32).reshape(-1, 1, 2) if len(
                                    filtered_new) > 0 else None
                        else:
                            p0 = p_new
                            spin_rates.append(spin_rates[-1] if spin_rates else 0.0)
                    else:
                        p0 = p_new
                        spin_rates.append(0.0)

                    old_gray = frame_gray.copy()

                    # Visualization
                    if visualization:
                        if p0 is not None:
                            for point in p0:
                                x, y = point.ravel()
                                cv.circle(vis_frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                        current_spin = spin_rates[-1] if spin_rates else 0.0
                        cv.putText(vis_frame, f"Spin: {current_spin:.1f} rad/s", (10, 30),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        frame_to_plot = cv.resize(vis_frame, dsize=(0, 0), fx=0.6, fy=0.6)
                        cv.imshow(Environment.CV_VISUALIZATION_NAME, frame_to_plot)
                        cv.waitKey(1)
                else:
                    # Ball is not detected in the frame
                    spin_rates.append(spin_rates[-1] if spin_rates else 0.0)

                frame_idx += 1

            # Save the spin rates for this camera
            spin_results[view.camera.name] = np.array(spin_rates, dtype=np.float32)
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)

        # Visualize ans save
        DataManager.save({"spin_results": spin_results, "axis_points": axis_points}, self.save_name)
        Environment.set("spin_rates", spin_results)
        Environment.set("axis_points", axis_points)
        self.plot_angular_speed(spin_results, graph_save_path, visualization)
        self.plot_axes(params, axis_points, graph_save_path, visualization)

    def load(self, params: dict):
        """
        Loads previously computed spin results.

        Args:
            params (dict): Dictionary containing optional flags such as
                - "visualization" (bool): Whether to display plots.

        Returns:
            dict: Dictionary containing:
                - "spin_results": Camera-wise spin rate arrays.
                - "axis_points": Camera-wise 3D rotation axes per frame.
        """

        # Load and save all parameters
        visualization = params.get("visualization", Environment.visualization)
        data = cast(dict, DataManager.load(self.save_name))
        spin_results = data.get("spin_results")
        axis_points = data.get("axis_points")
        Environment.set("spin_rates", spin_results)
        Environment.set("axis_points", axis_points)

        # Visualization
        if visualization:
            self.plot_angular_speed(spin_results, "", visualization)
            self.plot_axes(params, axis_points, "", visualization)

        input("\033[92mPress Enter to continue...\033[0m")

    def plotly_page(self, params: dict) -> dict[str, html.Div]:
        """
        Returns a Dash HTML Div with a 3D Plotly visualization of the ball trajectory
        and spin axes along the trajectory.

        Args:
            params (dict): Dictionary of visualization parameters, e.g.,
                - "radius" (float, optional): Ball radius in meters.

        Returns:
            dict[str, html.Div]: Mapping from class name to a Dash Div containing
                                 the interactive 3D figure.
        """

        # Load the previously saved 3D trajectory
        trajectory_3d = cast(BallTrajectory3d, Environment.get("3D_trajectory"))
        data = cast(dict, DataManager.load(self.save_name))
        axis_points = data.get("axis_points")

        # Helper function to create a 3D sphere at a given center with a specified radius
        def make_sphere(ball_center, ball_radius, resolution=10):
            u, v = np.mgrid[0: 2 * np.pi: resolution * 2j, 0: np.pi: resolution * 1j]
            return go.Surface(
                x=ball_radius * np.cos(u) * np.sin(v) + ball_center[0],
                y=ball_radius * np.sin(u) * np.sin(v) + ball_center[1],
                z=ball_radius * np.cos(v) + ball_center[2],
                opacity=0.6,
                showscale=False,
                name="Ball"
            )

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
        radius = params.get("ball_radius", Environment.ball_radius)

        # Add spin axes
        spin_axis_traces = []
        ball_diameter = radius * 2
        if axis_points:
            key = list(axis_points.keys())[0]
            for center, axis_vec in zip(xyz_t, axis_points[key]):
                if center is None or axis_vec is None:
                    continue
                axis_vec = np.array(axis_vec)
                if np.linalg.norm(axis_vec) < 1e-6:
                    continue
                axis_vec = axis_vec / np.linalg.norm(axis_vec)
                start = np.array(center) - axis_vec * ball_diameter
                end = np.array(center) + axis_vec * ball_diameter
                spin_axis_traces.append(
                    go.Scatter3d(
                        x=[start[0], end[0]],
                        y=[start[1], end[1]],
                        z=[start[2], end[2]],
                        mode="lines",
                        line=dict(width=3, color="red"),
                        opacity=0.4,
                        name="Spin Axis"
                    )
                )

        # Construct the figure
        lane = go.Figure(
            data=[
                go.Mesh3d(x=x, y=y, z=z, color="lightblue", opacity=0.8, name="Bowling Lane"),
                go.Scatter3d(x=x[1:3], y=y[1:3], z=z[1:3], mode="lines", name="Pit", line=dict(width=5, color="red")),
                go.Scatter3d(x=xt, y=yt, z=zt, mode="lines", name="Trajectory", line=dict(width=5, color="green")),
                *spin_axis_traces,
            ],
            layout=go.Layout(
                scene=dict(
                    xaxis=dict(title="X"),
                    yaxis=dict(title="Y"),
                    zaxis=dict(title="Z"),
                    aspectmode="data",
                    camera=dict(
                        eye=dict(x=-7.5, y=-1.07/2, z=1.5),
                        center=dict(x=20, y=-1.07/2, z=-10),
                    )
                ),
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
                title="Bowling lane with spin axes",
            ),
            frames=[go.Frame(data=make_sphere(pos, radius)) for pos in xyz_t],
        )

        graph = dcc.Graph(figure=lane, style={"width": "100%", "height": "95vh"})

        page = html.Div(children=graph)

        return {self.__class__.__name__: page}

    @staticmethod
    def plot_angular_speed(spin_results: dict, graph_save_path: str, visualization: bool):
        """
        Plots the ball spin rate over time for each camera.

        Args:
            spin_results (dict): Camera-wise spin rate arrays (rad/frame).
            graph_save_path (str): Directory to save the plot; empty string means no save.
            visualization (bool): Whether to display the plot interactively.
        """

        plt.figure(figsize=(10, 6))
        for cam_name, spins in spin_results.items():
            spins_rps = spins / (2 * np.pi)
            window = 5
            spins_smooth = np.convolve(spins_rps, np.ones(window) / window, mode='same')
            plt.plot(abs(spins_smooth), label=f"{cam_name}", alpha=0.5)

        plt.xlabel("Frame")
        plt.ylabel("Spin rate (rev/s)")
        plt.title("Ball Spin Rate Over Time")
        plt.legend()
        plt.grid(True)
        if graph_save_path != "":
            plt.savefig(
                f"{graph_save_path}/angular_speed/{Environment.save_name}_{Environment.video_name.removesuffix(".mp4")}.png",
                dpi=300, bbox_inches="tight")

        if visualization:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_axes(params: dict, axis_points: dict, graph_save_path: str, visualization: bool):
        """
        Plots the 3D rotation axes of the ball along its trajectory.

        Args:
            params (dict): Dictionary with optional visualization parameters (e.g., 'ball_radius').
            axis_points (dict): Camera-wise 3D rotation axes per frame.
            graph_save_path (str): Directory to save the plots; empty string means no save.
            visualization (bool): Whether to display the plots interactively.
        """

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Ball Spin Axes : 3D Visualization")

        plot_utils.bowling_lane(ax, np.array(Environment.coords["world_lane"]))
        plot_utils.trajectory(ax, cast(BallTrajectory3d, Environment.get("3D_trajectory")))

        ball_diameter = params.get("ball_radius", Environment.visualization) * 2
        trajectory_3d = Environment.get("3D_trajectory")

        cam_names = list(axis_points.keys())
        first_cam_name = cam_names[0]
        other_cam_name = cam_names[1] if len(cam_names) > 1 else None

        # Plot spin axes along the trajectory
        for i, center in enumerate(trajectory_3d.coords):
            if center[0] is None:
                continue

            axis_vec = axis_points[first_cam_name][i] if i < len(axis_points[first_cam_name]) else None
            if (axis_vec is None or np.linalg.norm(axis_vec) < 1e-6) and other_cam_name is not None:
                if i < len(axis_points[other_cam_name]):
                    axis_vec = axis_points[other_cam_name][i]
            if axis_vec is None or np.linalg.norm(axis_vec) < 1e-6:
                continue

            axis_vec = axis_vec / np.linalg.norm(axis_vec)
            start = np.array(center) - axis_vec * ball_diameter
            end = np.array(center) + axis_vec * ball_diameter
            ax.plot([start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    color='red', alpha=0.4)

        views = {"back": (20, 180), "top": (90, -90), "side": (0, 90), "front": (20, 0)}

        for name, (elev, azim) in views.items():
            ax.view_init(elev=elev, azim=azim)
            if graph_save_path != "":
                plt.savefig(
                    f"{graph_save_path}/axis/{Environment.save_name}_{name}_{Environment.video_name.removesuffix(".mp4")}.png",
                    dpi=300, bbox_inches="tight")

        if visualization:
            plt.show()
        else:
            plt.close(fig)
