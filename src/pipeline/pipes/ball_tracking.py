from collections.abc import Iterable
from typing import cast

from ultralytics import YOLO
import cv2 as cv
import dash_player as dp
import numpy as np
from dash import html
from pipeline.environment import BallTrajectory2d, DataManager, Environment
from pipeline.pipe import Pipe


class TrackBall(Pipe):
    """
    Tracks a bowling ball in video frames using YOLO object detection.

    This pipeline stage:
        1. Tracks the ball in each camera view.
        2. Saves the 2D trajectories for later pipeline stages.
        3. Optionally visualizes the tracking in real time.
        
    Supports side-by-side video visualization in Dash.
    """

    def execute(self, params: dict):
        """
        Tracks the ball in all camera views and saves the resulting trajectories.

        Args:
            params (dict): Dictionary containing:
                - "save_path" (str): Directory to save tracking results.
                - "visualization" (bool, optional): Whether to display tracking frames.
        """

        # Load the variables
        save_path = params["save_path"]
        visualization = params.get("visualization", Environment.visualization)

        tracking_results = []

        # Load the YOLO model for ball detection
        model = YOLO("./resources/models/yolov8l.pt")

        # Iterate over all camera views defined in the Environment
        for view in Environment.get_views():
            # Track the ball in the current view and store its trajectory
            view.trajectory = self.track_ball(model, view.video, visualization, save_path, view.camera.name)
            tracking_results.append({"name": view.camera.name, "trajectory": view.trajectory})

        # Save the results
        DataManager.save(tracking_results, self.save_name)

        input("\033[92mPress Enter to continue...\033[0m")

    def load(self, params: dict):
        """
        Loads previously tracked ball trajectories.

        Args:
            params (dict): Dictionary containing optional parameters:
                - "visualization" (bool): Whether to display video frames with tracked trajectory.
        """

        # Visualization
        try:
            visualization = params.get("visualization", Environment.visualization)
        except AttributeError as _:
            visualization = Environment.visualization

        # Load saved tracking results
        tracking_results = cast(Iterable, DataManager.load(self.save_name))

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
                frame2_resized = cv.resize(frame2, (int(frame2.shape[1] * (frame1.shape[0] / frame2.shape[0])),
                                                    frame1.shape[0]))
                stacked_frame = np.hstack((frame1, frame2_resized))
                frame_to_plot = cv.resize(stacked_frame, dsize=(0, 0), fx=0.5, fy=0.5)
                cv.imshow(Environment.CV_VISUALIZATION_NAME, frame_to_plot)
                cv.waitKey(1)
            cap1.set(cv.CAP_PROP_POS_FRAMES, 0)
            cap2.set(cv.CAP_PROP_POS_FRAMES, 0)

        input("\033[92mPress Enter to continue...\033[0m")

    def plotly_page(self, params: dict) -> dict[str, html.Div]:
        """
        Returns a Dash HTML Div containing side-by-side video players
        showing the tracked ball videos.

        Args:
            params (dict): Dictionary containing:
                - "save_path" (str): Directory where videos are stored for playback.

        Returns:
            dict[str, html.Div]: Mapping from class name to a Dash Div containing
                                 the interactive video players.
        """

        # Load the parameters
        save_path = params["save_path"]

        # Get first two views from Environment
        view1, view2 = Environment.get_views()[:2]
        folder = save_path.split("/")[-1]

        # Build URLs for Dash video player
        url1 = f"/video/{folder}/{view1.camera.name}/{Environment.save_name}_{Environment.video_name}"
        url2 = f"/video/{folder}/{view2.camera.name}/{Environment.save_name}_{Environment.video_name}"

        # Create HTML container for side-by-side layout
        page = html.Div(
            children=[
                dp.DashPlayer(
                    id="player-1",
                    url=url1,
                    controls=True,
                    width="100%",
                    height="100%",
                    loop=True,
                    playing=True,
                    style={"objectFit": "cover"}  # fill container, crop if needed
                ),
                dp.DashPlayer(
                    id="player-2",
                    url=url2,
                    controls=True,
                    width="100%",
                    height="100%",
                    loop=True,
                    playing=True,
                    style={"objectFit": "cover"}
                )
            ],
            style={
                "display": "flex",
                "flexDirection": "row",
                "height": "90vh"  # total available height
            }
        )

        return {self.__class__.__name__: page}

    @staticmethod
    def track_ball(model, video, visualization, save_path, camera_name) -> BallTrajectory2d:
        """
        Tracks a single ball in a video using YOLO detection and generates a 2D trajectory.

        Args:
            model: YOLO detection model instance.
            video: Video object containing a capture of frames.
            visualization (bool): Whether to display tracking frames in real time.
            save_path (str): Directory to save annotated tracking video and trajectory image.
            camera_name (str): Name of the camera (used for file naming).

        Returns:
            BallTrajectory2d: A 2D trajectory object containing the ball positions and radii
                              for all frames, interpolated where detections are missing.
        """

        # Reset video to the first frame
        video.capture.set(cv.CAP_PROP_POS_FRAMES, 0)
        tot_frames = int(video.capture.get(cv.CAP_PROP_FRAME_COUNT))
        width = int(video.capture.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(video.capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = video.capture.get(cv.CAP_PROP_FPS)

        # Initialize trajectory object
        trajectory = BallTrajectory2d(tot_frames)

        # Initialize the video writer if save_path is provided
        out_video = None
        if save_path:
            out_path = f"{save_path.replace("images", "videos")}/{camera_name}/{Environment.save_name}_{Environment.video_name}"
            out_video = cv.VideoWriter(out_path, cv.VideoWriter_fourcc(*'avc1'), fps, (width, height))

        last_box = None
        frame_idx = 0
        first_frame = None
        x1_crop, y1_crop, x2_crop, y2_crop = 0, 0, 0, 0

        while True:
            ret, frame = video.capture.read()
            if not ret:
                break

            if first_frame is None:
                first_frame = frame.copy()

            # Crop around last detection
            if last_box is None:
                yolo_frame = frame
                x1_offset, y1_offset = 0, 0
            else:
                x1, y1, x2, y2 = last_box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                w, h = int((x2 - x1) * 10), int((y2 - y1) * 10)
                x1_crop = max(cx - w // 2, 0)
                y1_crop = max(cy - h // 2, 0)
                x2_crop = min(cx + w // 2, frame.shape[1])
                y2_crop = min(cy + h // 2, frame.shape[0])
                yolo_frame = cv.resize(frame[y1_crop:y2_crop, x1_crop:x2_crop], (320, 320))
                x1_offset, y1_offset = x1_crop, y1_crop

            # YOLO detection
            results = model(yolo_frame, conf=0.05, classes=[32], augment=True, half=False)[0]

            ball_boxes = []
            for cls, box in zip(results.boxes.cls, results.boxes.xyxy):
                if last_box is not None:
                    scale_x = (x2_crop - x1_crop) / yolo_frame.shape[1]
                    scale_y = (y2_crop - y1_crop) / yolo_frame.shape[0]
                    x1b, y1b, x2b, y2b = box
                    ball_boxes.append([int(x1b * scale_x + x1_offset), int(y1b * scale_y + y1_offset),
                                       int(x2b * scale_x + x1_offset), int(y2b * scale_y + y1_offset)])
                else:
                    ball_boxes.append(list(map(int, box)))

            # Visualization
            vis_frame = frame.copy()
            if visualization:
                for b in ball_boxes:
                    cv.rectangle(vis_frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                trajectory.plot_onto(vis_frame)
                frame_to_plot = cv.putText(vis_frame.copy(), str(frame_idx), (5, 30),
                                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv.LINE_AA)
                cv.imshow(Environment.CV_VISUALIZATION_NAME, cv.resize(frame_to_plot, (0, 0), fx=0.6, fy=0.6))
                cv.waitKey(2)

            # Write frame to output video
            if out_video is not None:
                out_video.write(vis_frame)

            # Update trajectory
            if ball_boxes:
                x1, y1, x2, y2 = ball_boxes[0]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                radius = max(x2 - x1, y2 - y1) // 2
                trajectory.set_by_frame(np.array([cx, cy]), radius, frame_idx)
                last_box = [x1, y1, x2, y2]
            else:
                trajectory.set_by_frame(None, None, frame_idx)

            frame_idx += 1

        trajectory.interpolate_all()

        # Save trajectory image
        if save_path and first_frame is not None:
            full_traj_frame = first_frame.copy()
            trajectory.plot_onto(full_traj_frame)
            cv.imwrite(
                f"{save_path}/{camera_name}/{Environment.save_name}_{Environment.video_name.removesuffix(".mp4")}.png",
                full_traj_frame)

        if out_video is not None:
            out_video.release()

        return trajectory
