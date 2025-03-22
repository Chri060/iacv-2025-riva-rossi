import warnings

import cv2 as cv
import dash_player as dp
import librosa
import numpy as np
import scipy
from dash import html

from pipeline.environment import Environment, Video
from pipeline.pipe import Pipe

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class Synchronizer(Pipe):
    def execute(self, params: dict):
        # Load Parameters
        try:
            save_path = params["save_path"]
        except Exception as _:
            raise Exception("Missing required parameter : save_path")
        try:
            visualization = params.get("visualization", Environment.visualization)
        except AttributeError as _:
            visualization = Environment.visualization

        cam1_name = Environment.camera_names[0]
        cam2_name = Environment.camera_names[1]

        print(f"Synchronizing {cam1_name} & {cam2_name} videos")
        views = Environment.get_views()
        view1, view2 = views[0], views[1]

        # Compute output paths
        output1_path = f"{save_path}/{cam1_name}/{Environment.savename}_{Environment.video_names[0]}"
        output2_path = f"{save_path}/{cam2_name}/{Environment.savename}_{Environment.video_names[1]}"

        videos = [view1.video, view2.video]

        # Get the properties of the two videos (fps, duration, and resolution)
        fps1, duration1, _ = videos[0].get_video_properties()
        fps2, duration2, _ = videos[1].get_video_properties()

        # Select the target fps and duration as the lowest value between the two video
        target_fps = min(fps1, fps2)
        target_duration = min(duration1, duration2)

        # Sync the audio of the videos
        start_time = self.__get_time_shift(videos[0].path, videos[1].path)
        print(
            f"{cam1_name} is ~{int(abs(start_time))} seconds {'ahead' if start_time < 0 else 'behind'} of {cam2_name}"
        )

        print(
            f"Saving results ({round(target_duration, 2)} s | {round(target_fps, 2)} fps)"
        )
        # Process the videos to make them synchronized

        if start_time >= 0:
            self.__apply_shift(
                videos,
                output1_path,
                output2_path,
                target_fps,
                target_duration,
                start_time,
                0,
                visualization,
            )
        else:
            self.__apply_shift(
                videos,
                output1_path,
                output2_path,
                target_fps,
                target_duration,
                0,
                -start_time,
                visualization,
            )

        view1.video.capture.release()
        view2.video.capture.release()
        view1.video = Video(output1_path)
        view2.video = Video(output2_path)

    def load(self, params: dict):
        try:
            save_path = params["save_path"]
        except Exception as _:
            raise Exception("Missing required parameter : save_path")
        try:
            visualization = params.get("visualization", Environment.visualization)
        except AttributeError as _:
            visualization = Environment.visualization

        view = Environment.get_views()
        view1, view2 = view[0], view[1]

        output1_path = f"{save_path}/{view1.camera.name}/{Environment.savename}_{Environment.video_names[0]}"
        output2_path = f"{save_path}/{view2.camera.name}/{Environment.savename}_{Environment.video_names[1]}"

        if visualization:
            cap1 = cv.VideoCapture(output1_path)
            cap2 = cv.VideoCapture(output2_path)
            while cap1.isOpened() and cap2.isOpened():
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                if not (ret1 and ret2):
                    break
                frame2_resized = cv.resize(
                    frame2,
                    (
                        int(frame2.shape[1] * (frame1.shape[0] / frame2.shape[0])),
                        frame1.shape[0],
                    ),
                )
                stacked_frame = np.hstack((frame1, frame2_resized))
                stacked_frame = cv.resize(stacked_frame, dsize=(0, 0), fx=0.5, fy=0.5)
                cv.imshow(
                    Environment.CV_VISUALIZATION_NAME,
                    stacked_frame,
                )
                cv.waitKey(1)

        view1.video = Video(output1_path)
        view2.video = Video(output2_path)

    # Returns an estimated time shift in milliseconds between two video files based on audio correlation
    def __get_time_shift(self, video1_path: str, video2_path: str):
        # Extract the audio from the videos
        audio1, sr1 = librosa.load(video1_path, sr=None, mono=True)
        audio2, sr2 = librosa.load(video2_path, sr=None, mono=True)

        # Check if the audio sampling rates matches
        if sr1 != sr2:
            raise ValueError("Sampling rates do not match")

        # Compute the cross-correlation between the two audios
        corr = scipy.signal.correlate(audio1, audio2, mode="full")

        # Compute the lag between the two signals
        lag = np.argmax(corr) - (len(audio2) - 1)

        # Convert the lag in time difference
        time_shift = lag / sr1
        return time_shift

    def __apply_shift(
        self,
        videos: list[Video],
        output_path1: str,
        output_path2: str,
        target_fps: float,
        target_duration: float,
        start1: float,
        start2: float,
        visualization: bool,
    ):
        fourcc1 = cv.VideoWriter_fourcc(*"avc1")
        fourcc2 = cv.VideoWriter_fourcc(*"avc1")
        _, _, width_height1 = videos[0].get_video_properties()
        _, _, width_height2 = videos[1].get_video_properties()
        out1 = cv.VideoWriter(output_path1, fourcc1, target_fps, width_height1)
        out2 = cv.VideoWriter(output_path2, fourcc2, target_fps, width_height2)

        frame_time = 1 / target_fps
        frame_count = int(target_fps * target_duration)
        count = 0
        current_time1 = start1
        current_time2 = start2

        captures = [videos[0].capture, videos[1].capture]

        while captures[0].isOpened() and captures[1].isOpened() and count < frame_count:
            captures[0].set(cv.CAP_PROP_POS_MSEC, current_time1 * 1000)
            captures[1].set(cv.CAP_PROP_POS_MSEC, current_time2 * 1000)
            ret1, frame1 = captures[0].read()
            ret2, frame2 = captures[1].read()
            if not (ret1 and ret2):
                break
            if visualization:
                frame2_resized = cv.resize(
                    frame2,
                    (
                        int(frame2.shape[1] * (frame1.shape[0] / frame2.shape[0])),
                        frame1.shape[0],
                    ),
                )
                stacked_frame = np.hstack((frame1, frame2_resized))
                stacked_frame = cv.putText(
                    stacked_frame,
                    str(count),
                    (30, 30),
                    cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(255, 0, 0),
                    thickness=2,
                    lineType=cv.LINE_AA,
                )
                stacked_frame = cv.resize(stacked_frame, dsize=(0, 0), fx=0.5, fy=0.5)
                cv.imshow(Environment.CV_VISUALIZATION_NAME, stacked_frame)
                cv.waitKey(1)
            out1.write(frame1)
            out2.write(frame2)
            count += 1
            current_time1 += frame_time
            current_time2 += frame_time

    def plotly_page(self, params: dict) -> dict[str, html.Div]:
        try:
            save_path = params["save_path"]
        except Exception as _:
            raise Exception("Missing required parameter : save_path")

        view = Environment.get_views()
        view1, view2 = view[0], view[1]

        folder = save_path.split("/")[-1]

        url1 = f"/video/{folder}/{view1.camera.name}/{Environment.savename}_{Environment.video_names[0]}"
        url2 = f"/video/{folder}/{view2.camera.name}/{Environment.savename}_{Environment.video_names[1]}"

        dp1 = dp.DashPlayer(
            id="player-1",
            url=url1,
            controls=True,
            width="100%",
            loop=True,
            playing=True,
        )
        dp2 = dp.DashPlayer(
            id="player-2",
            url=url2,
            controls=True,
            width="100%",
            loop=True,
            playing=True,
        )

        page = html.Div(
            children=[
                html.Div(
                    children=dp1,
                    style={"heigth": "auto", "width": "49%", "display": "inline-block"},
                ),
                html.Div(
                    children=dp2,
                    style={"heigth": "auto", "width": "49%", "display": "inline-block"},
                ),
            ]
        )
        return {self.__class__.__name__: page}


class Stabilizer(Pipe):
    def execute(self, params: dict):
        return super().execute(params)

    def load(self, params: dict):
        return super().load(params)

    def plotly_page(self, params: dict):
        return None


class Undistorcer(Pipe):
    def execute(self, params: dict):
        try:
            save_path = params["save_path"]
        except Exception as _:
            raise Exception("Missing required parameter : save_path")
        try:
            visualization = params.get("visualization", Environment.visualization)
        except AttributeError as _:
            visualization = Environment.visualization

        for i, view in enumerate(Environment.get_views()):
            cam = view.camera
            cap = view.video.capture
            fps, duration, width_height = view.video.get_video_properties()
            output_path = f"{save_path}/{cam.name}/{Environment.savename}_{Environment.video_names[i]}"
            fourcc = cv.VideoWriter_fourcc(*"avc1")
            out = cv.VideoWriter(output_path, fourcc, fps, width_height)
            new_intrinsic, _ = cv.getOptimalNewCameraMatrix(
                cam.intrinsic, cam.distortion, width_height, 1, width_height
            )
            while view.video.capture.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_undistort = cv.undistort(
                    frame,
                    cam.intrinsic,
                    cam.distortion,
                    None,
                    newCameraMatrix=new_intrinsic,
                )
                if visualization:
                    frame_to_plot = cv.resize(
                        frame_undistort, dsize=(0, 0), fx=0.5, fy=0.5
                    )
                    cv.imshow(Environment.CV_VISUALIZATION_NAME, frame_to_plot)
                    cv.waitKey(1)
                out.write(frame_undistort)
            cap.release()
            out.release()
            view.video = Video(output_path)
            cam.intrinsic = new_intrinsic

    def load(self, params: dict):
        try:
            save_path = params["save_path"]
        except Exception as _:
            raise Exception("Missing required parameter : save_path")
        try:
            visualization = params["visualization"]
        except Exception as _:
            visualization = Environment.visualization

        paths = []
        for i, camera_name in enumerate(Environment.camera_names):
            view = Environment.get(camera_name)
            path = f"{save_path}/{camera_name}/{Environment.savename}_{Environment.video_names[i]}"
            paths.append(path)
            view.video.capture.release()
            view.video = Video(path)

        if visualization:
            cap1 = cv.VideoCapture(paths[0])
            cap2 = cv.VideoCapture(paths[1])
            while cap1.isOpened() and cap2.isOpened():
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                if not ret1 or not ret2:
                    break
                frame2_resized = cv.resize(
                    frame2,
                    (
                        int(frame2.shape[1] * (frame1.shape[0] / frame2.shape[0])),
                        frame1.shape[0],
                    ),
                )
                stacked_frame = np.hstack((frame1, frame2_resized))
                frame_to_plot = cv.resize(stacked_frame, dsize=(0, 0), fx=0.5, fy=0.5)
                cv.imshow(Environment.CV_VISUALIZATION_NAME, frame_to_plot)
                cv.waitKey(1)

    def plotly_page(self, params: dict) -> dict[str, html.Div]:
        try:
            save_path = params["save_path"]
        except Exception as _:
            raise Exception("Missing required parameter : save_path")

        view = Environment.get_views()
        view1, view2 = view[0], view[1]

        folder = save_path.split("/")[-1]

        url1 = f"/video/{folder}/{view1.camera.name}/{Environment.savename}_{Environment.video_names[0]}"
        url2 = f"/video/{folder}/{view2.camera.name}/{Environment.savename}_{Environment.video_names[1]}"

        dp1 = dp.DashPlayer(
            id="player-1",
            url=url1,
            controls=True,
            width="100%",
            loop=True,
            playing=True,
        )
        dp2 = dp.DashPlayer(
            id="player-2",
            url=url2,
            controls=True,
            width="100%",
            loop=True,
            playing=True,
        )

        page = html.Div(
            children=[
                html.Div(
                    children=dp1,
                    style={"heigth": "auto", "width": "49%", "display": "inline-block"},
                ),
                html.Div(
                    children=dp2,
                    style={"heigth": "auto", "width": "49%", "display": "inline-block"},
                ),
            ]
        )
        return {self.__class__.__name__: page}
