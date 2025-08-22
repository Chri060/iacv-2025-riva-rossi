from typing import cast

import cv2 as cv
import numpy as np
from cv2.typing import MatLike
from pipeline.environment import DataManager, Environment
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
            params (dict): Dictionary that may contain the 'scale' key for image scaling
                           during manual point selection (default: [0.7, 0.7]).
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
        No plotly implementation.
        """

        return None

    @staticmethod
    def manual_point_selection(image: MatLike, save_path: str, camera_name: str, scale: float = 0.5):
        """
        Allows the user to manually select points on an image using mouse clicks.
        Points are scaled back to original image coordinates and returned as a list.
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
