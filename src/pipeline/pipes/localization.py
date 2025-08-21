import random

from ultralytics import YOLO
import cv2 as cv
import dash_player as dp
import numpy as np
import plotly.graph_objects as go
from cv2.typing import MatLike
from dash import dcc, html
from scipy.interpolate import make_smoothing_spline
import matplotlib.pyplot as plt
import pipeline.plot_utils as plot_utils
from pipeline.environment import BallTrajectory2d, BallTrajectory3d, DataManager, Environment
from pipeline.pipe import Pipe

class DetectLane(Pipe):
    """
        DetectLane is a pipeline stage responsible for detecting and storing lane corner points
        in each camera view. The detection relies on manual point selection, where the user
        clicks on lane corners within sampled frames from the input videos.
    """
    def execute(self, params: dict):
        """
        Executes lane detection for all camera views.
        For each camera:
          - Reads the middle frame of the video.
          - Allows manual selection of lane corners.
          - Stores the selected corners.
        Finally, saves all detection results using DataManager.

        Args:
            params (dict): Dictionary that may contain 'scale' key for image scaling
                           during manual point selection (default: [0.7, 0.7]).
        """

        # Scale factor
        try:
            scales = params.get("scale", [0.7, 0.7])
        except Exception as _:
            scales = [0.7, 0.7]

        detection_results = {}

        # Iterate over all camera views defined in the Environment
        for i, view in enumerate(Environment.get_views()):
            # Get video duration from video properties
            _, duration, _ = view.video.get_video_properties()
            capture = view.video.capture

            # Move capture to the middle frame of the video
            capture.set(cv.CAP_PROP_POS_MSEC, duration * 1000 / 2)

            # Read the selected frame
            ret, frame = capture.read()

            # Manually select lane corner points on the chosen frame
            view.lane.corners = np.array(self.__manual_point_selection(frame, scale=scales[i]))

            # Reset video capture to the first frame
            capture.set(cv.CAP_PROP_POS_FRAMES, 0)

            # Store detected corners for this view
            detection_results.update({view.camera.name: view.lane.corners})

        # Save results
        DataManager.save(detection_results, self.save_name)

    def load(self, params: dict):
        """
        Loads previously detected lane corner points and optionally visualizes them.
        If visualization is enabled, draws the points on the middle frame of each video
        and displays stacked side-by-side frames.

        Args:
            params (dict): Dictionary that may contain 'visualization' key to enable/disable
                           visualization (default: Environment.visualization).
        """

        # Visualization
        try:
            visualization = params.get("visualization", Environment.visualization)
        except AttributeError as _:
            visualization = Environment.visualization

        # Load previously saved detection results
        detection_results = DataManager.load(self.save_name)

        frames = []

        # Iterate over all camera views
        for view in Environment.get_views():
            # Assign stored corners back to the view
            view.lane.corners = detection_results[view.camera.name]

            if visualization:
                capture = view.video.capture
                _, duration, _ = view.video.get_video_properties()

                # Read the middle frame again
                capture.set(cv.CAP_PROP_POS_MSEC, duration * 1000 / 2)
                ret, frame = capture.read()
                capture.set(cv.CAP_PROP_POS_MSEC, 0)

                # Draw each corner point on the frame
                for point in view.lane.corners:
                    # Draw a larger red circle
                    frame = cv.circle(frame, (int(point[0]), int(point[1])), 6, (0, 0, 255), -1)  # Red filled circle
                    # Optional: black border
                    frame = cv.circle(frame, (int(point[0]), int(point[1])), 6, (0, 0, 0), 1)

                frames.append(frame)

        if visualization:
            frame1 = frames[0]
            frame2 = frames[1]
            frame2_resized = cv.resize(frame2, (int(frame2.shape[1] * (frame1.shape[0] / frame2.shape[0])), frame1.shape[0]))
            stacked_frame = np.hstack((frame1, frame2_resized))
            frame_to_plot = cv.resize(stacked_frame, dsize=(0, 0), fx=0.5, fy=0.5)
            cv.imshow(Environment.CV_VISUALIZATION_NAME, frame_to_plot)
            cv.waitKey(2000)

    @staticmethod
    def __manual_point_selection(image: MatLike, scale: float = 0.5):
        """
        Allows the user to manually select points on an image using mouse clicks.
        Points are scaled back to original image coordinates and returned as a list.

        Args:
            image (MatLike): Input image on which points are selected.
            scale (float): Scaling factor to resize image for easier point selection
                           (default: 0.5).

        Returns:
            List[List[float]]: List of selected points [[x1, y1], [x2, y2], ...].
        """

        selected_points = []
        upscale = 1 / scale

        # Mouse callback to handle point selection
        def select_point(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                # Draw a small red dot where the user clicks
                cv.circle(image, (x, y), 3, (0, 0, 255), -1)
                cv.imshow(Environment.CV_VISUALIZATION_NAME, image)

                # Rescale coordinates back to the original image size
                x = x * upscale
                y = y * upscale
                selected_points.append([x, y])

        # Scale down the frame for easier manual selection
        image = cv.resize(image, None, fx=scale, fy=scale)

        # Show the image
        cv.imshow(Environment.CV_VISUALIZATION_NAME, image)

        # Set the mouse callback to capture clicks
        cv.setMouseCallback(Environment.CV_VISUALIZATION_NAME, select_point)

        # Wait until the user presses Enter (key code 13)
        key = cv.waitKey(0) & 0xFF
        if key == 13:
            return selected_points

    @staticmethod
    def plotly_page() -> None:
        return None

class TrackBall(Pipe):
    """
    TrackBall is a pipeline stage responsible for loading, tracking, and optionally visualizing
    ball trajectories from videos. Supports Dash integration for web-based visualization.
    """

    def execute(self, params: dict):
        """
        Executes ball tracking using YOLO and saves the trajectories.
        """

        # Save path
        try:
            save_path = params["save_path"]
        except Exception as _:
            raise Exception("Missing required parameter : save_path")

        # Visualization
        try:
            visualization = params.get("visualization", Environment.visualization)
        except AttributeError as _:
            visualization = Environment.visualization

        tracking_results = []


        # Load the YOLO model for ball detection
        model = YOLO("./resources/models/yolov8l.pt")

        # Iterate over all camera views defined in the Environment
        for view in Environment.get_views():
            # Track the ball in the current view and store its trajectory
            view.trajectory = self.__track_ball(model, view.video, visualization)
            tracking_results.append({"name": view.camera.name, "trajectory": view.trajectory})

        # Save the results
        DataManager.save(tracking_results, self.save_name)

    def load(self, params: dict):
        """
        Loads previously tracked ball trajectories from storage and optionally visualizes them.
        """

        # Visualization
        try:
            visualization = params.get("visualization", Environment.visualization)
        except AttributeError as _:
            visualization = Environment.visualization

        # Load saved tracking results
        tracking_results = DataManager.load(self.save_name)

        captures, trajectories = [], []

        # Assign trajectories back to the corresponding views
        for result in tracking_results:
            view = Environment.get(result["name"])
            view.trajectory = result["trajectory"]
            if visualization:
                # Collect captures and trajectories for visualization
                captures.append(view.video.capture)
                trajectories.append(view.trajectory)

        if visualization and captures:
            cap1, cap2 = captures[:2]
            tr1, tr2 = trajectories[:2]
            while True:
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                if not ret1 or not ret2:
                    break
                tr1.plot_onto(frame1)
                tr2.plot_onto(frame2)
                frame2_resized = cv.resize(frame2, (int(frame2.shape[1] * (frame1.shape[0] / frame2.shape[0])), frame1.shape[0]))
                stacked_frame = np.hstack((frame1, frame2_resized))
                frame_to_plot = cv.resize(stacked_frame, dsize=(0, 0), fx=0.5, fy=0.5)
                cv.imshow(Environment.CV_VISUALIZATION_NAME, frame_to_plot)
                cv.waitKey(1)
            cap1.set(cv.CAP_PROP_POS_FRAMES, 0)
            cap2.set(cv.CAP_PROP_POS_FRAMES, 0)

    def plotly_page(self, params: dict) -> dict[str, html.Div]:
        """
        Creates a Dash page to visualize tracked ball videos side by side.
        """

        # Save path
        try:
            save_path = params["save_path"]
        except Exception as _:
            raise Exception("Missing required parameter : save_path")

        # Get first two views from Environment
        view1, view2 = Environment.get_views()[:2]
        folder = save_path.split("/")[-1]

        # Build URLs for Dash video player
        url1 = f"/video/{folder}/{view1.camera.name}/{Environment.save_name}_{Environment.video_names[0]}"
        url2 = f"/video/{folder}/{view2.camera.name}/{Environment.save_name}_{Environment.video_names[1]}"

        # Create DashPlayer components for each video
        dp1 = dp.DashPlayer(id="player-1", url=url1, controls=True, width="100%", loop=True, playing=True)
        dp2 = dp.DashPlayer(id="player-2", url=url2, controls=True, width="100%", loop=True, playing=True)

        # Create HTML container for side-by-side layout
        page = html.Div(
            children=[
                html.Div(children=dp1, style={"heigth": "auto", "width": "49%", "display": "inline-block"}),
                html.Div(children=dp2, style={"heigth": "auto", "width": "49%", "display": "inline-block"})
            ]
        )

        return {self.__class__.__name__: page}

    @staticmethod
    def __track_ball(model, video, visualization) -> BallTrajectory2d:
        """
        Tracks the ball in a video using YOLO, returning a 2D trajectory.
        """

        # Reset video to the first frame
        video.capture.set(cv.CAP_PROP_POS_FRAMES, 0)
        tot_frames = int(video.capture.get(cv.CAP_PROP_FRAME_COUNT))

        # Initialize trajectory object for storing ball positions per frame
        trajectory = BallTrajectory2d(tot_frames)

        last_box = None
        frame_idx = 0

        while True:
            # Read the next frame
            ret, frame = video.capture.read()
            if not ret:
                break

            # Crop around last detection for faster YOLO processing
            if last_box is None:
                # No previous detection, use full frame
                yolo_frame = frame
                x1_offset, y1_offset = 0, 0
            else:
                # Use last detected bounding box to crop region of interest
                x1, y1, x2, y2 = last_box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                w, h = int((x2 - x1) * 10), int((y2 - y1) * 10)
                x1_crop = max(cx - w // 2, 0)
                y1_crop = max(cy - h // 2, 0)
                x2_crop = min(cx + w // 2, frame.shape[1])
                y2_crop = min(cy + h // 2, frame.shape[0])

                # Resize cropped region to YOLO input size
                yolo_frame = cv.resize(frame[y1_crop:y2_crop, x1_crop:x2_crop], (320, 320))
                x1_offset, y1_offset = x1_crop, y1_crop

            # Run YOLO on the current frame or crop
            results = model(yolo_frame, conf=0.05, classes=[32], augment=True)[0]

            ball_boxes = []

            # Process YOLO detection results
            for cls, box in zip(results.boxes.cls, results.boxes.xyxy):
                if last_box is not None:
                    # Rescale coordinates from cropped frame back to original full frame
                    # Rescale box coordinates back to full frame
                    scale_x = (x2_crop - x1_crop) / yolo_frame.shape[1]
                    scale_y = (y2_crop - y1_crop) / yolo_frame.shape[0]
                    x1b, y1b, x2b, y2b = box
                    ball_boxes.append([int(x1b * scale_x + x1_offset), int(y1b * scale_y + y1_offset), int(x2b * scale_x + x1_offset), int(y2b * scale_y + y1_offset)])
                else:
                    # Full-frame detection, use original coordinates
                    ball_boxes.append(list(map(int, box)))

            # Visualize
            if visualization:
                vis_frame = frame.copy()
                for b in ball_boxes:
                    cv.rectangle(vis_frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                trajectory.plot_onto(vis_frame)
                frame_to_plot = cv.resize(vis_frame, dsize=(0, 0), fx=0.6, fy=0.6)
                frame_to_plot = cv.putText(frame_to_plot, str(frame_idx), (5, 30),
                                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv.LINE_AA)
                cv.imshow(Environment.CV_VISUALIZATION_NAME, frame_to_plot)
                cv.waitKey(2)

            # Update trajectory with detected ball position
            if ball_boxes:
                x1, y1, x2, y2 = ball_boxes[0]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                radius = max(x2 - x1, y2 - y1) // 2
                trajectory.set_by_frame(np.array([cx, cy]), radius, frame_idx)
                last_box = [x1, y1, x2, y2]
            else:
                # No ball detected in this frame
                trajectory.set_by_frame(None, None, frame_idx)

            frame_idx += 1

        # Interpolate missing trajectory points (frames without detection)
        trajectory.interpolate_all()

        return trajectory

class LocalizeBall(Pipe):
    """
    Class to localize a bowling ball in 3D space using stereo camera views.
    """

    def execute(self, params: dict):
        """
        Main execution method that computes the 3D trajectory of the ball.

        Args:
            params (dict): Parameters dict. Can include 'visualization' to enable 3D plotting.
        """

        # Get the visualization flag from params or default to environment setting
        try:
            visualization = params.get("visualization", False)
        except Exception as _:
            visualization = Environment.visualization

        # Retrieve the two camera views from the environment
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
        lam = 100
        t = np.arange(0, len(points_3d), 1)
        y1 = points_3d[:, 0]
        y2 = points_3d[:, 1]
        y3 = points_3d[:, 2]
        spl_x = make_smoothing_spline(t, y1, lam=lam)
        spl_y = make_smoothing_spline(t, y2, lam=lam)
        spl_z = make_smoothing_spline(t, y3, lam=lam)

        # Visualize
        if visualization:
           plot_utils.plot_3d_spline_interpolation(t, points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], spl_x(t), spl_y(t), spl_z(t))

        # Update points_3d with smoothed coordinates
        points_3d = np.array([spl_x(t), spl_y(t), spl_z(t)]).T.reshape(-1, 3)

        # Store the 3D trajectory in a Ball_Trajectory_3D object
        trajectory_3d = BallTrajectory3d(views[0].trajectory.n_frames)
        start_3d = start + start_over
        end_3d = start + end_over
        for i, frame in enumerate(range(start_3d, end_3d)):
            trajectory_3d.set_by_frame(points_3d[i], frame)

        # Save the trajectory
        Environment.set("3D_trajectory", trajectory_3d)
        DataManager.save(trajectory_3d, self.save_name)

        if visualization:
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title("Ball Localization : 3D Visualization")

            # Plot lane and trajectory
            plot_utils.bowling_lane(ax, np.array(Environment.coords["world_lane"]))
            plot_utils.trajectory(ax, Environment.get("3D_trajectory"))

            # Define views: (elev, azim)
            views = {
                "front": (20, 0),
                "back": (20, 180),
                "top": (90, -90),
                "lateral": (0, 90)
            }

            for name, (elev, azim) in views.items():
                ax.view_init(elev=elev, azim=azim)
                plt.savefig(f"ball_trajectory_{name}.png")
                print(f"Saved {name} view as ball_trajectory_{name}.png")

            plt.show()

    def load(self, params: dict):
        trajectory_3d = DataManager.load(self.save_name)
        Environment.set("3D_trajectory", trajectory_3d)
        return

    def plotly_page(self, params: dict) -> dict[str, html.Div]:
        """
        Returns a Dash HTML Div with a 3D plotly visualization of the trajectory and lane.
        Includes animation of the ball along its trajectory.
        """

        # Load the previously saved 3D trajectory
        trajectory_3d = DataManager.load(self.save_name)

        # Helper function to create a 3D sphere at a given center with a specified radius
        def make_sphere(center, radius, resolution=10):
            x, y, z = center
            u, v = np.mgrid[0 : 2 * np.pi : resolution * 2j, 0 : np.pi : resolution * 1j]
            X = radius * np.cos(u) * np.sin(v) + x
            Y = radius * np.sin(u) * np.sin(v) + y
            Z = radius * np.cos(v) + z
            return go.Surface(x=X, y=Y, z=Z, opacity=0.6)

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
        r = 0.108

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
                                ],  # Stops animation
                            },
                        ],
                    }
                ],
                title="Bowling Lane",
            ),
            frames=[go.Frame(data=make_sphere(pos, r)) for pos in xyz_t],
        )

        lane.update_scenes(aspectmode="data")

        graph = dcc.Graph(figure=lane, style={"width": "100%", "height": "95vh"})

        page = html.Div(children=graph)

        return {self.__class__.__name__: page}

class SpinBall(Pipe):
    """
    SpinBall is a pipeline stage responsible for estimating bowling ball spin
    using optical flow on previously tracked ball trajectories.
    """

    def execute(self, params: dict):
        """
        Estimates ball spin using weighted optical flow on tracked ball regions
        and plots 2D rotation axis.
        """

        # Visualization flag
        try:
            visualization = params.get("visualization", Environment.visualization)
        except AttributeError:
            visualization = Environment.visualization

        smoothing_alpha = 0.2
        max_corners = 50

        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        spin_results = {}
        axis_points = {}

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

            while True:
                ret, frame = cap.read()
                if not ret or frame_idx >= trajectory.n_frames:
                    break

                frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                vis_frame = frame.copy()
                cxcy, radius = trajectory.get_by_frame(frame_idx)

                if cxcy is not None and radius is not None:
                    cx, cy = map(int, cxcy)
                    radius = int(radius)

                    if visualization:
                        cv.circle(vis_frame, (cx, cy), radius, (0, 0, 255), 2)
                        cv.circle(vis_frame, (cx, cy), 3, (0, 0, 255), -1)

                    # Mask ball area for feature tracking
                    mask_ball = np.zeros_like(frame_gray)
                    cv.circle(mask_ball, (cx, cy), max(radius - 2, 1), 255, -1)
                    p_new = cv.goodFeaturesToTrack(frame_gray, mask=mask_ball, maxCorners=max_corners, qualityLevel=0.01, minDistance=5)

                    if p0 is not None and len(p0) > 0:
                        p1, st, _ = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                        if p1 is not None and st is not None:
                            good_new = p1[st == 1]
                            good_old = p0[st == 1]

                            filtered_new, filtered_old = [], []
                            for (new, old) in zip(good_new, good_old):
                                x_new, y_new = new.ravel()
                                if (x_new - cx) ** 2 + (y_new - cy) ** 2 <= (radius - 2) ** 2:
                                    filtered_new.append(new)
                                    filtered_old.append(old)

                            trajectory_3d = Environment.get("3D_trajectory")

                            # --- Compute 3D rotation axis & spin ---
                            axes_3d = []
                            dthetas = []
                            weights = []

                            for (new, old) in zip(filtered_new, filtered_old):
                                # Old & new displacement vectors in *image plane*
                                dx_old, dy_old = old.ravel() - [cx, cy]
                                dx_new, dy_new = new.ravel() - [cx, cy]

                                r_old_len = np.sqrt(dx_old ** 2 + dy_old ** 2)
                                if r_old_len < 0.3 * radius:
                                    continue

                                # Normalize to sphere surface in 3D
                                # Map 2D offsets into 3D local coords (z from sphere geometry)
                                z_old = np.sqrt(max(radius ** 2 - r_old_len ** 2, 0.0))
                                z_new = np.sqrt(max(radius ** 2 - (dx_new ** 2 + dy_new ** 2), 0.0))

                                r_old = np.array([dx_old, dy_old, z_old])
                                r_new = np.array([dx_new, dy_new, z_new])

                                # Rotation axis from cross product
                                axis_vec = np.cross(r_old, r_new)
                                if np.linalg.norm(axis_vec) > 1e-6:
                                    axis_vec /= np.linalg.norm(axis_vec)
                                    axes_3d.append(axis_vec)

                                # Rotation angle from dot product
                                dot = np.dot(r_old, r_new) / (np.linalg.norm(r_old) * np.linalg.norm(r_new))
                                dot = np.clip(dot, -1.0, 1.0)
                                theta = np.arccos(dot)

                                dthetas.append(theta)
                                weights.append(r_old_len / radius)

                            if dthetas:
                                avg_axis = np.mean(axes_3d, axis=0) if axes_3d else np.array([0, 0, 1])
                                avg_axis /= np.linalg.norm(avg_axis)
                            else:
                                avg_axis = np.array([0, 0, 1])

                            # Save 3D axis instead of 2D line endpoints
                            axis_points[view.camera.name].append(avg_axis)

                            if dthetas:
                                med_dtheta = np.average(dthetas, weights=weights)
                                spin_rate = med_dtheta * fps
                                prev_spin = spin_rates[-1] if spin_rates else 0.0
                                if len(spin_rates) > 0 and abs(spin_rate - prev_spin) > 20:
                                    spin_rate = prev_spin
                                spin_rate = smoothing_alpha * spin_rate + (1 - smoothing_alpha) * prev_spin
                                spin_rates.append(spin_rate)
                            else:
                                spin_rates.append(spin_rates[-1] if spin_rates else 0.0)

                            # Merge features
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
                    spin_rates.append(spin_rates[-1] if spin_rates else 0.0)

                frame_idx += 1

            spin_results[view.camera.name] = np.array(spin_rates, dtype=np.float32)
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)

        DataManager.save(spin_results, self.save_name)
        Environment.set("spin_rates", spin_results)
        Environment.set("axis_points", axis_points)

        plt.figure(figsize=(10, 6))
        for cam_name, spins in spin_results.items():
            spins_rps = spins / (2 * np.pi)
            window = 5
            spins_smooth = np.convolve(spins_rps, np.ones(window) / window, mode='same')
            plt.plot(abs(spins_smooth), label=f"{cam_name}")

        plt.xlabel("Frame")
        plt.ylabel("Spin rate (rev/s)")
        plt.title("Ball Spin Rate Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig("spin_rate_over_time.png", dpi=300, bbox_inches="tight")
        plt.show()

        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # --- Known real ball radius (in same units as trajectory_3d) ---
        R_ball = 0.11  # meters (example: bowling ball ~11 cm radius)

        # --- Find the first valid center ---
        valid_centers = [c for c in trajectory_3d.coords if c[0] is not None]
        if not valid_centers:
            raise ValueError("No valid 3D coordinates found in trajectory.")

        center0 = valid_centers[0]

        # --- Draw reference sphere at the first valid center ---
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = center0[0] + R_ball * np.outer(np.cos(u), np.sin(v))
        y = center0[1] + R_ball * np.outer(np.sin(u), np.sin(v))
        z = center0[2] + R_ball * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.3, linewidth=0)

        # --- Plot all predicted axes ---
        L = R_ball * 2  # axis length scaling
        for frame_idx, (center, axis_vec) in enumerate(zip(trajectory_3d.coords, axis_points[view.camera.name])):
            if center[0] is None:
                continue  # skip missing centers
            if axis_vec is None or np.linalg.norm(axis_vec) < 1e-6:
                continue
            axis_vec = axis_vec / np.linalg.norm(axis_vec)
            start = center - axis_vec * L
            end = center + axis_vec * L
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                    color='red', alpha=0.3)

        # --- Formatting ---
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Ball Spin Axes Over Time")
        ax.set_box_aspect([1, 1, 1])  # equal aspect ratio

        plt.show()

    def load(self, params: dict):
        """
        Load previously saved spin data.

        Args:
            params (dict): Dictionary that may contain the 'visualization' flag
        """
        return

    def plotly_page(self, params: dict) -> dict[str, html.Div]:
        return