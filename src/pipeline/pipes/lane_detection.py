from typing import cast

import cv2 as cv
import numpy as np
from cv2.typing import MatLike
from pipeline.environment import DataManager, Environment
from pipeline.pipe import Pipe


class DetectLane(Pipe):
    """
    Detects and stores lane corner points in camera views using manual point selection.

    This pipeline stage:
        1. Reads the middle frame of each camera video.
        2. Allows the user to manually click on lane corners.
        3. Stores the selected corner coordinates in the Environment.
        4. Optionally visualizes the detected corners on the video frames.
        5. Saves all detection results using DataManager.
    """

    def execute(self, params: dict):
        """
        Executes lane detection for all camera views.

        For each camera reads the middle frame of the video, allows manual selection of lane corners,
        and stores the selected corners in the Environment.
        Finally, saves all detection results using DataManager.

        Args:
            params (dict): Configuration dictionary. Can contain:
                save_path (str): Directory where annotated frames will be saved.
                scale (float | list[float], optional): Scaling factor(s) for display during manual selection.
                    Defaults to [0.7, 0.7].
        """

        # Load parameters
        save_path = params["save_path"]
        scales = params.get("scale", [0.7, 0.7])

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
            view.lane.corners = np.array(
                self.manual_point_selection(frame, save_path, view.camera.name, scale=scales[i]))

            # Reset video capture to the first frame
            capture.set(cv.CAP_PROP_POS_FRAMES, 0)

            # Store detected corners for this view
            detection_results.update({view.camera.name: view.lane.corners})

        # Save results
        DataManager.save(detection_results, self.save_name)

        input("\033[92mPress Enter to continue...\033[0m")

    def load(self, params: dict):
        """
        Loads previously detected lane corner points.

        If visualization is enabled, draws the points on the middle frame of each video
        and displays stacked side-by-side frames.

        Args:
            params (dict): Configuration dictionary. Can contain:
                visualization (bool, optional): Whether to visualize the loaded points.
                    Defaults to Environment.visualization.
        """

        # Load parameters
        visualization = params.get("visualization", Environment.visualization)
        detection_results = cast(dict, DataManager.load(self.save_name))

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
            frame2_resized = cv.resize(frame2,
                                       (int(frame2.shape[1] * (frame1.shape[0] / frame2.shape[0])), frame1.shape[0]))
            stacked_frame = np.hstack((frame1, frame2_resized))
            frame_to_plot = cv.resize(stacked_frame, dsize=(0, 0), fx=0.5, fy=0.5)
            cv.imshow(Environment.CV_VISUALIZATION_NAME, frame_to_plot)
            cv.waitKey(2000)

        input("\033[92mPress Enter to continue...\033[0m")

    def plotly_page(self, params: dict):
        """
        Placeholder for a Plotly visualization page. Not implemented.
        """

        return None

    @staticmethod
    def manual_point_selection(image: MatLike, save_path: str, camera_name: str, scale: float = 0.5):
        """
        Allows the user to manually select points on an image using mouse clicks.

        Selected points are scaled back to original image coordinates and returned as a list.

        Args:
            image (MatLike): Input BGR image.
            save_path (str): Directory to save the annotated image.
            camera_name (str): Name of the camera (used for filename).
            scale (float, optional): Scale factor for display during selection. Defaults to 0.5.

        Returns:
            list[list[float]] | None: List of selected points as [x, y] in original image coordinates.
                Returns None if no points are selected.
        """

        selected_points = []
        upscale = 1 / scale

        # Make a copy for display
        display_image = cv.resize(image.copy(), None, fx=scale, fy=scale)

        def select_point(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                # Draw on the scaled display image
                cv.circle(display_image, (x, y), 3, (0, 0, 255), -1)
                cv.imshow(Environment.CV_VISUALIZATION_NAME, display_image)

                # Append coordinates scaled back to the original
                selected_points.append([x * upscale, y * upscale])

        cv.imshow(Environment.CV_VISUALIZATION_NAME, display_image)
        cv.setMouseCallback(Environment.CV_VISUALIZATION_NAME, select_point)

        # Wait for Enter
        key = cv.waitKey(0) & 0xFF
        if key == 13 and selected_points:
            # Draw selected points on the original full-size image
            for pt in selected_points:
                cv.circle(image, (int(pt[0]), int(pt[1])), 15, (0, 0, 255), -1)

            final_save_path = f"{save_path}/{camera_name}/{Environment.save_name}_{Environment.video_name.removesuffix('.mp4')}.png"
            cv.imwrite(final_save_path, image)
            cv.destroyAllWindows()
            return selected_points

        cv.destroyAllWindows()
        return None
