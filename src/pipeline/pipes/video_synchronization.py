import warnings, scipy, librosa
import cv2 as cv
import dash_player as dp
import numpy as np
from dash import html
from pipeline.environment import Environment, Video
from pipeline.pipe import Pipe

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class SynchronizeVideo(Pipe):
    """
    Synchronizes two video streams using their audio tracks.

    This pipeline stage:
        1. Computes the time offset between two videos by cross-correlating audio.
        2. Applies the calculated shift to produce synchronized video streams.
        3. Updates the Environment with the synchronized Video objects.
        4. Optionally visualizes synchronization side by side in OpenCV.
        5. Supports side-by-side Dash video visualization for web display.
    """

    def execute(self, params: dict):
        """
        Executes the video synchronization process.

        Args:
            params (dict): Dictionary containing parameters:
                save_path (str): Path to save the synchronized videos. Required.
                visualization (bool, optional): Whether to visualize synchronization. Defaults to Environment.visualization.
        """

        # Get the parameters
        save_path = params["save_path"]
        visualization = params.get("visualization", Environment.visualization)

        # Get names of the cameras from the Environment
        cam1_name = Environment.camera_names[0]
        cam2_name = Environment.camera_names[1]

        print(f"Synchronizing {cam1_name} & {cam2_name} videos")

        # Get video views from the environment
        views = Environment.get_views()
        view1, view2 = views[0], views[1]

        # Compute output paths for saving synchronized videos
        output1_path = f"{save_path}/{cam1_name}/{Environment.save_name}_{Environment.video_name}"
        output2_path = f"{save_path}/{cam2_name}/{Environment.save_name}_{Environment.video_name}"

        # Retrieve the video objects
        videos = [view1.video, view2.video]

        # Get video properties: fps, duration, and resolution
        fps1, duration1, _ = videos[0].get_video_properties()
        fps2, duration2, _ = videos[1].get_video_properties()

        # Use the minimum fps and duration to ensure synchronization
        target_fps = min(fps1, fps2)
        target_duration = min(duration1, duration2)

        # Compute the audio time shift between the two videos
        start_time = self.get_time_shift(videos[0].path, videos[1].path)
        print(
            f"{cam1_name} is ~{int(abs(start_time))} seconds {'ahead' if start_time < 0 else 'behind'} of {cam2_name}")

        print(f"Saving results ({round(target_duration, 2)} s | {round(target_fps, 2)} fps)")

        # Apply the calculated time shift to synchronize videos
        if start_time >= 0:
            self.apply_shift(videos, output1_path, output2_path, target_fps, target_duration, start_time, 0,
                             visualization)
        else:
            self.apply_shift(videos, output1_path, output2_path, target_fps, target_duration, 0, -start_time,
                             visualization)

        # Release original video captures and replace with synchronized versions
        view1.video.capture.release()
        view2.video.capture.release()
        view1.video = Video(output1_path)
        view2.video = Video(output2_path)

        input("\n\033[92mPress Enter to continue...\033[0m")

    def load(self, params: dict):
        """
        Loads and optionally visualizes previously synchronized videos.

        Args:
            params (dict): Dictionary containing parameters:
                save_path (str): Path where synchronized videos are saved. Required.
                visualization (bool, optional): Whether to show videos side by side. Defaults to Environment.visualization.
        """

        # Get the parameters
        save_path = params["save_path"]
        visualization = params.get("visualization", Environment.visualization)

        view = Environment.get_views()
        view1, view2 = view[0], view[1]

        # Compute paths to the saved synchronized videos
        output1_path = f"{save_path}/{view1.camera.name}/{Environment.save_name}_{Environment.video_name}"
        output2_path = f"{save_path}/{view2.camera.name}/{Environment.save_name}_{Environment.video_name}"

        # Show videos side by side
        if visualization:
            cap1 = cv.VideoCapture(output1_path)
            cap2 = cv.VideoCapture(output2_path)
            while cap1.isOpened() and cap2.isOpened():
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                if not (ret1 and ret2):
                    break
                frame2_resized = cv.resize(frame2, (int(frame2.shape[1] * (frame1.shape[0] / frame2.shape[0])),
                                                    frame1.shape[0]))
                stacked_frame = np.hstack((frame1, frame2_resized))
                stacked_frame = cv.resize(stacked_frame, dsize=(0, 0), fx=0.5, fy=0.5)
                cv.imshow(Environment.CV_VISUALIZATION_NAME, stacked_frame)
                cv.waitKey(1)

        # Load the videos into the view objects
        view1.video = Video(output1_path)
        view2.video = Video(output2_path)

        input("\033[92mPress Enter to continue...\033[0m")

    def plotly_page(self, params: dict) -> dict[str, html.Div]:
        """
        Generates a Dash page displaying the synchronized videos side by side.

        Args:
            params (dict): Dictionary containing parameters:
                save_path (str): Path where synchronized videos are saved.

        Returns:
            dict[str, html.Div]: Mapping of class name to the Dash HTML page containing video players.
        """

        # Get the parameters
        save_path = params["save_path"]

        # Get the two views for display
        view = Environment.get_views()
        view1, view2 = view[0], view[1]

        # Extract the folder name from the save path
        folder = save_path.split("/")[-1]

        # Build URLs for DashPlayer video components
        url1 = f"/video/{folder}/{view1.camera.name}/{Environment.save_name}_{Environment.video_name}"
        url2 = f"/video/{folder}/{view2.camera.name}/{Environment.save_name}_{Environment.video_name}"

        # Arrange players side by side in a single Div
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
                    style={"objectFit": "cover"}
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
                "height": "90vh"
            }
        )

        return {self.__class__.__name__: page}

    @staticmethod
    def get_time_shift(video1_path: str, video2_path: str) -> float:
        """
        Computes the time offset between two videos using their audio tracks.

        Args:
            video1_path (str): Path to the first video.
            video2_path (str): Path to the second video.

        Returns:
            float: The time-shift in seconds. Positive if video1 is ahead of video2.
        """
        try:
            # Load audio from videos
            audio1, sr1 = librosa.load(video1_path, sr=None, mono=True)
            audio2, sr2 = librosa.load(video2_path, sr=None, mono=True)

            # Resample both audios to the same sample rate
            target_sr = min(sr1, sr2)
            if sr1 != target_sr:
                audio1 = librosa.resample(audio1, orig_sr=sr1, target_sr=target_sr)
            if sr2 != target_sr:
                audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=target_sr)

            # Normalize audio signals
            audio1 = (audio1 - np.mean(audio1)) / (np.std(audio1) + 1e-8)
            audio2 = (audio2 - np.mean(audio2)) / (np.std(audio2) + 1e-8)

            # Cross-correlate audio to find lag
            corr = scipy.signal.correlate(audio1, audio2, mode="full")
            lag = np.argmax(corr) - (len(audio2) - 1)

            # Convert lag to time in seconds
            time_shift = lag / target_sr

            return time_shift

        except Exception as e:
            print(f"Error processing audio: {e}")
            return 0.0

    @staticmethod
    def apply_shift(videos: list[Video], output_path1: str, output_path2: str, target_fps: float,
                    target_duration: float, start1: float, start2: float, visualization: bool):
        """
        Applies time shift to videos and saves the synchronized versions.

        Args:
            videos (list[Video]): List of two Video objects to synchronize.
            output_path1 (str): Path to save the first synchronized video.
            output_path2 (str): Path to save the second synchronized video.
            target_fps (float): Frame rate for output videos.
            target_duration (float): Duration for output videos in seconds.
            start1 (float): Start time offset for the first video in seconds.
            start2 (float): Start time offset for the second video in seconds.
            visualization (bool): Whether to visualize the synchronization process in OpenCV.
        """

        # Initialize video writers
        fourcc1 = cv.VideoWriter_fourcc(*"avc1")
        fourcc2 = cv.VideoWriter_fourcc(*"avc1")

        # Get width and height of both videos
        _, _, width_height1 = videos[0].get_video_properties()
        _, _, width_height2 = videos[1].get_video_properties()

        # Create OpenCV VideoWriter objects to save output videos
        out1 = cv.VideoWriter(output_path1, fourcc1, target_fps, width_height1)
        out2 = cv.VideoWriter(output_path2, fourcc2, target_fps, width_height2)

        # Calculate frame time and total number of frames to process
        frame_time = 1 / target_fps
        frame_count = int(target_fps * target_duration)
        count = 0

        # Initialize current playback times based on offsets
        current_time1 = start1
        current_time2 = start2

        # Access OpenCV capture objects from the Video instances
        captures = [videos[0].capture, videos[1].capture]

        # Process each frame until the end or target frame count
        while captures[0].isOpened() and captures[1].isOpened() and count < frame_count:
            # Set the capture positions for both videos according to their offsets
            captures[0].set(cv.CAP_PROP_POS_MSEC, current_time1 * 1000)
            captures[1].set(cv.CAP_PROP_POS_MSEC, current_time2 * 1000)

            # Read frames from both videos
            ret1, frame1 = captures[0].read()
            ret2, frame2 = captures[1].read()
            if not (ret1 and ret2):
                break

            # Side-by-side with frame index visualization
            if visualization:
                frame2_resized = cv.resize(frame2, (int(frame2.shape[1] * (frame1.shape[0] / frame2.shape[0])),
                                                    frame1.shape[0]))
                stacked_frame = np.hstack((frame1, frame2_resized))
                stacked_frame = cv.putText(stacked_frame, str(count), (30, 30), cv.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                           color=(255, 0, 0), thickness=2, lineType=cv.LINE_AA)
                stacked_frame = cv.resize(stacked_frame, dsize=(0, 0), fx=0.5, fy=0.5)
                cv.imshow(Environment.CV_VISUALIZATION_NAME, stacked_frame)
                cv.waitKey(1)

            # Write frames to output videos
            out1.write(frame1)
            out2.write(frame2)

            # Increment frame counters and time offsets
            count += 1
            current_time1 += frame_time
            current_time2 += frame_time
