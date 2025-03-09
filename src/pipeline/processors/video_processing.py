from pipeline.environment import Environment
from pipeline.environment import Video
from pipeline.pipe import Pipe
import cv2 as cv
import librosa
import scipy
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class Synchronizer(Pipe):
    def execute(self, params):
        # Load Parameters
        try:
            save_path = params["save_path"]
        except Exception as _:
            raise Exception("Missing required parameter : save_path")
        visualization = params.get("visualization", Environment.visualization)

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

        if visualization:
            cv.destroyAllWindows()

        view1.video.capture.release()
        view2.video.capture.release()
        view1.video = Video(output1_path)
        view2.video = Video(output2_path)

    def load(self, params):
        try:
            save_path = params["save_path"]
        except Exception as _:
            raise Exception("Missing required parameter : save_path")
        visualization = params.get("visualization", Environment.visualization)

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
                    f"Synchronization of {view1.camera.name} and {view2.camera.name}",
                    stacked_frame,
                )
                cv.waitKey(1)
            cv.destroyAllWindows()

        view1.video = Video(output1_path)
        view2.video = Video(output2_path)

    # Returns an estimated time shift in milliseconds between two video files based on audio correlation
    def __get_time_shift(self, video1_path, video2_path):
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
        videos,
        output_path1,
        output_path2,
        target_fps,
        target_duration,
        start1,
        start2,
        visualization,
    ):
        fourcc1 = cv.VideoWriter_fourcc(*"mp4v")
        fourcc2 = cv.VideoWriter_fourcc(*"mp4v")
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
                cv.imshow("sync", stacked_frame)
                cv.waitKey(1)
            out1.write(frame1)
            out2.write(frame2)
            count += 1
            current_time1 += frame_time
            current_time2 += frame_time


class Stabilizer(Pipe):
    def execute(self, params):
        return super().execute(params)

    def load(self, params):
        return super().load(params)


class Undistorcer(Pipe):
    def execute(self, params):
        try:
            save_path = params["save_path"]
        except Exception as _:
            raise Exception("Missing required parameter : save_path")
        visualization = params.get("visualization", Environment.visualization)

        for i, view in enumerate(Environment.get_views()):
            cam = view.camera
            cap = view.video.capture
            fps, duration, width_height = view.video.get_video_properties()
            output_path = f"{save_path}/{cam.name}/{Environment.savename}_{Environment.video_names[i]}"
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
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
                    cv.imshow(cam.name, frame_undistort)
                    cv.waitKey(1)
                out.write(frame_undistort)
            if visualization:
                cv.destroyAllWindows()
            cap.release()
            out.release()
            view.video = Video(output_path)
            cam.intrinsic = new_intrinsic

    def load(self, params):
        try:
            save_path = params["save_path"]
        except:
            raise Exception("Missing required parameter : save_path")
        try:
            visualization = params["visualization"]
        except:
            visualization = Environment.visualization

        for i, camera_name in enumerate(Environment.camera_names):
            view = Environment.get(camera_name)
            path = f"{save_path}/{camera_name}/{Environment.savename}_{Environment.video_names[i]}"
            view.video.capture.release()
            if visualization:
                cap = cv.VideoCapture(path)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cv.imshow(f"Undistorcer : {camera_name}", frame)
                    cv.waitKey(1)
                cv.destroyAllWindows()
            view.video = Video(path)
