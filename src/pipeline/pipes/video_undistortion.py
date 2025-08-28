import warnings
import cv2 as cv
import dash_player as dp
import numpy as np
from dash import html
from pipeline.environment import Environment, Video
from pipeline.pipe import Pipe

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class UndistortVideo(Pipe):
    """
    Undistorts videos using camera intrinsic and distortion parameters.

    This pipeline stage:
        1. Reads all videos from the Environment.
        2. Applies undistortion to each frame using known calibration parameters.
        3. Saves the undistorted videos to the specified path.
        4. Updates the Environment with the new undistorted Video objects.
        5. Optionally visualizes undistorted frames in real-time via OpenCV.
    
    Supports side-by-side Dash video visualization for web display.
    """

    def execute(self, params: dict):
        """
        Executes undistortion on all videos in the environment.

        Args:
            params (dict): Dictionary containing:
                save_path (str): Directory to save undistorted videos. Required.
                visualization (bool, optional): Whether to visualize frames during processing.
                    Defaults to Environment.visualization.
        """

        # Get the parameters
        save_path = params["save_path"]
        visualization = params.get("visualization", Environment.visualization)

        # Process each view (camera and video pair) in the environment
        for i, view in enumerate(Environment.get_views()):
            cam = view.camera
            cap = view.video.capture
            fps, duration, width_height = view.video.get_video_properties()

            # Define the output video path
            output_path = f"{save_path}/{cam.name}/{Environment.save_name}_{Environment.video_name}"

            # Prepare the video writer
            fourcc = cv.VideoWriter_fourcc(*"avc1")
            out = cv.VideoWriter(output_path, fourcc, fps, width_height)

            # Compute optimal intrinsic matrix (removes black borders)
            new_intrinsic, _ = cv.getOptimalNewCameraMatrix(cam.intrinsic, cam.distortion, width_height, 1,
                                                            width_height)

            # Process all frames in the video
            while view.video.capture.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Undistort frame
                frame_undistort = cv.undistort(frame, cam.intrinsic, cam.distortion, None,
                                               newCameraMatrix=new_intrinsic)

                # Real-time display of undistorted video
                if visualization:
                    frame_to_plot = cv.resize(frame_undistort, dsize=(0, 0), fx=0.5, fy=0.5)
                    cv.imshow(Environment.CV_VISUALIZATION_NAME, frame_to_plot)
                    cv.waitKey(1)

                # Write undistorted frame to output video
                out.write(frame_undistort)

            # Release resources
            cap.release()
            out.release()

            # Update the environment with new video and camera intrinsics
            view.video = Video(output_path)
            cam.intrinsic = new_intrinsic

        input("\033[92mPress Enter to continue...\033[0m")

    def load(self, params: dict):
        """
        Loads previously undistorted videos into the environment.

        Optionally displays the videos side by side using OpenCV.

        Args:
            params (dict): Dictionary containing:
                save_path (str): Directory of saved undistorted videos. Required.
                visualization (bool, optional): Whether to visualize after loading.
                    Defaults to Environment.visualization.
        """

        # Load the parameters
        save_path = params["save_path"]
        visualization = params.get("visualization", Environment.visualization)

        paths = []

        # Reload undistorted videos into each camera view
        for i, camera_name in enumerate(Environment.camera_names):
            view = Environment.get(camera_name)
            path = f"{save_path}/{camera_name}/{Environment.save_name}_{Environment.video_name}"

            # Replace old capture with undistorted one
            paths.append(path)
            view.video.capture.release()
            view.video = Video(path)

        # Show both undistorted videos side by side
        if visualization:
            cap1 = cv.VideoCapture(paths[0])
            cap2 = cv.VideoCapture(paths[1])
            while cap1.isOpened() and cap2.isOpened():
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                if not ret1 or not ret2:
                    break
                frame2_resized = cv.resize(frame2, (int(frame2.shape[1] * (frame1.shape[0] / frame2.shape[0])),
                                                    frame1.shape[0]))
                stacked_frame = np.hstack((frame1, frame2_resized))
                frame_to_plot = cv.resize(stacked_frame, dsize=(0, 0), fx=0.5, fy=0.5)
                cv.imshow(Environment.CV_VISUALIZATION_NAME, frame_to_plot)
                cv.waitKey(1)

        input("\033[92mPress Enter to continue...\033[0m")

    def plotly_page(self, params: dict) -> dict[str, html.Div]:
        """
        Generates a Dash page displaying undistorted videos side by side.

        Args:
            params (dict): Dictionary containing:
                save_path (str): Directory containing undistorted videos. Required.

        Returns:
            dict[str, html.Div]: Mapping of class name to the Dash HTML page containing video players.
        """

        # Get the parameters
        save_path = params["save_path"]

        # Get the first two views
        view = Environment.get_views()
        view1, view2 = view[0], view[1]

        # Folder name
        folder = save_path.split("/")[-1]

        # Construct URLs for both videos
        url1 = f"/video/{folder}/{view1.camera.name}/{Environment.save_name}_{Environment.video_name}"
        url2 = f"/video/{folder}/{view2.camera.name}/{Environment.save_name}_{Environment.video_name}"

        # Layout side by side
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
