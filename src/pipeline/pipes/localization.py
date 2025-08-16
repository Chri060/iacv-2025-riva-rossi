import math

from ultralytics import YOLO

import cv2 as cv
import dash_player as dp
import numpy as np
import plotly.graph_objects as go
from cv2.typing import MatLike
from dash import dcc, html
from numpy.typing import NDArray
from scipy.interpolate import make_smoothing_spline
from sklearn.kernel_ridge import KernelRidge

import pipeline.plot_utils as plot_utils
from pipeline.environment import (
    Ball_Trajectory_2D,
    Ball_Trajectory_3D,
    DataManager,
    Environment,
    Video,
)
from pipeline.pipe import Pipe


class Lane_Detector(Pipe):
    def execute(self, params: dict):
        try:
            scales = params.get("scale", [0.7, 0.7])
        except Exception as _:
            scales = [0.7, 0.7]
        detection_results = {}
        for i, view in enumerate(Environment.get_views()):
            _, duration, _ = view.video.get_video_properties()
            capture = view.video.capture
            # reads the frame in the middle of the video
            capture.set(cv.CAP_PROP_POS_MSEC, duration * 1000 / 2)
            ret, frame = capture.read()
            view.lane.corners = np.array(
                self.__manual_point_selection(frame, scale=scales[i])
            )
            capture.set(cv.CAP_PROP_POS_FRAMES, 0)
            detection_results.update({view.camera.name: view.lane.corners})
        DataManager.save(detection_results, self.save_name)

    def load(self, params: dict):
        try:
            visualization = params.get("visualization", Environment.visualization)
        except AttributeError as _:
            visualization = Environment.visualization

        detection_results = DataManager.load(self.save_name)
        frames = []
        for view in Environment.get_views():
            view.lane.corners = detection_results[view.camera.name]
            if visualization:
                capture = view.video.capture
                _, duration, _ = view.video.get_video_properties()
                # reads the frame in the middle of the video
                capture.set(cv.CAP_PROP_POS_MSEC, duration * 1000 / 2)
                ret, frame = capture.read()
                capture.set(cv.CAP_PROP_POS_MSEC, 0)
                for point in view.lane.corners:
                    frame = cv.circle(
                        frame, (int(point[0]), int(point[1])), 2, (255, 255, 255), 5
                    )
                    frame = cv.circle(
                        frame, (int(point[0]), int(point[1])), 2, (0, 0, 0), 1
                    )
                frames.append(frame)

        if visualization:
            frame1 = frames[0]
            frame2 = frames[1]
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
            cv.waitKey(2000)

    def __manual_point_selection(self, image: MatLike, scale: float = 0.5):
        selected_points = []
        upscale = 1 / scale

        def select_point(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                cv.circle(image, (x, y), 3, (0, 0, 255), -1)
                cv.imshow(Environment.CV_VISUALIZATION_NAME, image)
                x = x * upscale
                y = y * upscale
                selected_points.append([x, y])

        image = cv.resize(image, None, fx=scale, fy=scale)
        cv.imshow(Environment.CV_VISUALIZATION_NAME, image)
        cv.setMouseCallback(Environment.CV_VISUALIZATION_NAME, select_point)

        while True:
            if cv.waitKey(0) & 0xff == ord("q"):
                return selected_points


    def plotly_page(self, params: dict) -> None:
        return None

class Ball_Tracker(Pipe):
    def execute(self, params: dict):
        pass

    def load(self, params: dict):
        try:
            visualization = params.get("visualization", Environment.visualization)
        except AttributeError as _:
            visualization = Environment.visualization

        tracking_results = DataManager.load(self.save_name)
        captures = []
        trajectories = []
        for result in tracking_results:
            view = Environment.get(result["name"])
            view.trajectory = result["trajectory"]

            if visualization:
                captures.append(view.video.capture)
                trajectories.append(view.trajectory)

        if visualization:
            cap1 = captures[0]
            cap2 = captures[1]
            tr1 = trajectories[0]
            tr2 = trajectories[1]
            while True:
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                if not ret1 or not ret2:
                    break
                tr1.plot_onto(frame1)
                tr2.plot_onto(frame2)
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

            cap1.set(cv.CAP_PROP_POS_FRAMES, 0)
            cap2.set(cv.CAP_PROP_POS_FRAMES, 0)


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

class Ball_Tracker_Hough(Ball_Tracker):
    def execute(self, params:  dict):
        try:
            save_path = params.get("save_path")
        except Exception as _:
            raise Exception("Missing required parameter : save_path")
        try:
            visualization = params.get("visualization", Environment.visualization)
        except AttributeError as _:
            visualization = Environment.visualization

        tracking_results = []

        for i, view in enumerate(Environment.get_views()):
            video = view.video
            roi = view.lane.corners
            output_path = f"{save_path}/{view.camera.name}/{Environment.savename}_{Environment.video_names[i]}"
            trajectory = self.__track_moving_object(
                video,
                roi,
                kernel_size=5,
                playback_scaling=0.6,
                visualization=visualization,
            )

            # interpolate the roughly found trajectory
            trajectory = self.__interpolate(trajectory, visualization)

            # store the trajectory into the view
            view.trajectory = trajectory

            self.__save_tracking_video(video, trajectory, output_path, visualization)

            tracking_results.append(
                {"name": view.camera.name, "trajectory": view.trajectory}
            )


        print(tracking_results)

        # Save the final results
        DataManager.save(tracking_results, self.save_name)

    def __track_moving_object(
        self,
        video: Video,
        roi_points: NDArray,
        kernel_size: int = 20,
        color_range: list[tuple, tuple] = [(0, 0, 0), (255, 255, 255)],
        playback_scaling: float = 0.5,
        visualization: bool = False,
    ):
        if not video.capture.isOpened():
            print("Could not open the video!")
            return

        # Initialize video playback
        frame_counter = -1  # first frame is 0
        video.capture.set(cv.CAP_PROP_POS_FRAMES, 0)

        # Initialize the background subtractor
        bg_subtractor = cv.createBackgroundSubtractorMOG2(
            history=300, varThreshold=15, detectShadows=True
        )

        # Initialize found trajectories
        trajectories: set[Ball_Trajectory_2D] = set()

        # Get video properties and initialize polygonal bounding box
        _, _, width_height = video.get_video_properties()
        tot_frames = int(video.capture.get(cv.CAP_PROP_FRAME_COUNT))
        width, height = width_height
        mask_poly = self.__create_polygonal_mask(roi_points, 20, (height, width, 3))

        # Process each frame
        while True:
            ret, frame = video.capture.read()
            if not ret:
                break

            frame_counter += 1

            # Apply polygonal mask
            frame_poly = cv.bitwise_and(frame, mask_poly)

            def enhance_contrast(frame):
                lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
                l_channel, a, b = cv.split(lab)
                clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l_channel)
                limg = cv.merge((cl, a, b))
                return cv.cvtColor(limg, cv.COLOR_LAB2BGR)

            def suppress_reflections(frame):
                # Convert to grayscale and apply a threshold to detect bright spots
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                _, reflection_mask = cv.threshold(gray, 240, 255, cv.THRESH_BINARY)
                reflection_mask = cv.dilate(reflection_mask, np.ones((5, 5), np.uint8))
                frame[reflection_mask == 255] = 0  # Set reflection areas to black
                return frame

            frame_poly = enhance_contrast(frame_poly)
            frame_poly = suppress_reflections(frame_poly)


            # Apply background subtraction
            mask_fg = bg_subtractor.apply(frame_poly)

            # Remove MOG2 detected shadows
            mask_fg_no_shadows = mask_fg.copy()
            mask_fg_no_shadows[mask_fg == 127] = 0

            # Clean the mask using morphological operations and apply the mask
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask_morph = cv.morphologyEx(mask_fg_no_shadows, cv.MORPH_OPEN, kernel)
            frame_morph = cv.bitwise_and(frame_poly, frame_poly, mask=mask_morph)

            # Convert to HSV and apply color masking if needed
            frame_hsv = cv.cvtColor(frame_morph, cv.COLOR_BGR2HSV)
            mask_color = cv.inRange(frame_hsv, color_range[0], color_range[1])

            # Apply morphological opening on the color mask
            mask_color = cv.morphologyEx(mask_color, cv.MORPH_OPEN, kernel)

            # Combine color mask and final frame
            mask_color = cv.cvtColor(mask_color, cv.COLOR_GRAY2BGR)
            frame_result = cv.bitwise_and(frame_morph, mask_color)

            # Obtain the gray image + Gaussian blur for Hough Circles detection
            # Preprocessing: apply a Gaussian blur
            frame_result_gray = cv.cvtColor(frame_result, cv.COLOR_BGR2GRAY)
            # frame_result_gray = cv.GaussianBlur(
            #     frame_result_gray, (9, 9), 2
            # )  # Stronger blur to reduce noise
            circles = cv.HoughCircles(
                frame_result_gray,
                cv.HOUGH_GRADIENT,
                dp=1.05,  # Inverse ratio of resolution
                minDist=100,  # Minimum distance between circles
                param1=45,  # Canny edge threshold
                param2=17,  # Circle detection threshold
                minRadius=2,  # Minimum radius
                maxRadius=70,  # Maximum radius
            )

            if circles is not None:  # update trajectories
                trajectories = self.__update_trajectories(
                    trajectories, circles[0, :], frame_counter, tot_frames
                )
            else:  # keep the positions for each trajectory
                for trajectory in trajectories:
                    xy, r = trajectory.get_by_frame(frame_counter - 1)
                    trajectory.set_by_frame(xy, r, frame_counter)

            if visualization:
                for trajectory in trajectories:
                    trajectory.plot_onto(frame_poly)

                frame_to_plot = cv.resize(
                    frame_poly, dsize=(0, 0), fx=playback_scaling, fy=playback_scaling
                )

                frame_to_plot = cv.putText(
                    frame_to_plot,
                    str(frame_counter),
                    (5, 30),
                    cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 0),
                    thickness=1,
                    lineType=cv.LINE_AA,
                )
                frame_to_plot = cv.putText(
                    frame_to_plot,
                    f"{len(trajectories)} trajectories",
                    (5, 50),
                    cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 0),
                    thickness=1,
                    lineType=cv.LINE_AA,
                )

                cv.imshow(Environment.CV_VISUALIZATION_NAME, frame_to_plot)
                cv.waitKey(2)

        # Reroll video
        video.capture.set(cv.CAP_PROP_POS_FRAMES, 0)

        return self.__choose_trajectory(trajectories)

    # Returns the longest trajectory from the proposed ones
    def __choose_trajectory(self, trajectories: list[Ball_Trajectory_2D]):
        best_trajectory = None
        oldest_start = float("inf")
        for trajectory in trajectories:
            if trajectory.start < oldest_start:
                oldest_start = trajectory.start
                best_trajectory = trajectory
        return best_trajectory

    # Updates the ongoing trajectories discarding the one that cannot be continued
    def __update_trajectories(
        self,
        curr_trajectories: list[Ball_Trajectory_2D],
        circles: list[tuple[int]],
        curr_frame: int,
        tot_frames: int,
    ):
        to_assign = set(curr_trajectories.copy())
        res_trajectories = set()
        while len(circles) > 0:
            best_couple: dict[str, object] | None = None
            if (
                len(to_assign) > 0
            ):  # find the best trajectory-circle couple and assign that to the trajectory
                for trajectory in to_assign:
                    best_circle = None
                    min_dist = float("inf")
                    last_xy, _ = trajectory.get_by_frame(curr_frame - 1)  # last
                    for circle in circles:
                        x, y, r = circle
                        dist = math.hypot(x - last_xy[0], y - last_xy[1])
                        if dist < min_dist:
                            min_dist = dist
                            best_circle = circle
                    if best_couple is None:
                        best_couple = {
                            "traj": trajectory,
                            "circle": best_circle,
                            "dist": min_dist,
                        }
                    elif min_dist < best_couple["dist"]:
                        best_couple = {
                            "traj": trajectory,
                            "circle": best_circle,
                            "dist": min_dist,
                        }

                # place the best match into the trajectory and add that to the results
                x, y, r = best_couple["circle"]
                best_couple["traj"].set_by_frame(np.array((x, y)), r, curr_frame)
                res_trajectories.add(best_couple["traj"])

                # remove the trajectory
                to_assign.remove(best_couple["traj"])
                # remove the circle
                circles = circles[~np.all(circles == best_couple["circle"], axis=1)]
            else:  # if there are no more trajectories but there are still circles it creates new trajectories
                for circle in circles:
                    new_trajectory = Ball_Trajectory_2D(tot_frames)
                    x, y, r = circle
                    new_trajectory.set_by_frame(np.array((x, y)), r, curr_frame)
                    res_trajectories.add(new_trajectory)
                break
        return res_trajectories

    # Smoothes out noise from trajectory and detected radiuses
    def __interpolate(self, trajectory: Ball_Trajectory_2D, visualization: bool):
        coords = trajectory.get_coords()
        radiuses = trajectory.get_radiuses()
        start, end = trajectory.start, trajectory.end

        # Spline Interpolation for the trajectory
        lam = 100
        t = np.arange(start, end, 1)
        y1 = coords[:, 0]
        y2 = coords[:, 1]
        spl_x = make_smoothing_spline(t, y1, lam=lam)
        spl_y = make_smoothing_spline(t, y2, lam=lam)

        if visualization:
            plot_utils.plot_2d_spline_interpolation(t, y1, y2, spl_x(t), spl_y(t))

        # Ridge Regression with the Kernel Trick using additive chi2 as a kernel for the radius
        t = np.arange(start, end).reshape(-1, 1)
        y = radiuses.reshape(-1, 1)

        alpha = 1
        radius_predictor = KernelRidge(alpha=alpha, kernel="additive_chi2")
        radius_predictor.fit(t, y)

        new_radiuses = radius_predictor.predict(t)

        if visualization:
            plot_utils.plot_regression(
                x=t.reshape(-1),
                y_train=radiuses,
                y_pred=new_radiuses,
                title="Radius regression",
                xlabel="Frames",
                ylabel="Radius"
            )

        # Creating a new trajectory
        new_trajectory = Ball_Trajectory_2D(trajectory.n_frames)
        for frame in t:
            new_trajectory.set_by_frame(
                np.array((spl_x(frame)[0], spl_y(frame)[0])),
                radius_predictor.predict([frame]),
                curr_frame=frame[0],
            )

        return new_trajectory

    # Creates a binary polygonal mask of a given size
    def __create_polygonal_mask(self, corners: NDArray, padding: int, size: int):
        polygon_corners = np.array(corners, dtype=np.int32)

        # Create a black mask and fill the polygon
        mask = np.zeros(size[:2], dtype=np.uint8)
        cv.fillPoly(mask, [polygon_corners], 255)

        # Apply dilation with an elliptic kernel to smooth the edges
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (padding * 1, padding * 5))
        mask = cv.dilate(mask, kernel, iterations=1)

        if size[2] == 3:  # makes the mask 3 channel if the input size was a 3 channel
            mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        return mask

    # Stores a video containing only the tracked part of the image
    def __save_tracking_video(
        self,
        video: Video,
        trajectory: Ball_Trajectory_2D,
        output_path: str,
        visualization: bool,
    ):
        # Reroll video
        video.capture.set(cv.CAP_PROP_POS_FRAMES, 0)

        # Get video properties and initialize polygonal bounding box
        fps, _, width_height = video.get_video_properties()

        fourcc1 = cv.VideoWriter_fourcc(*"mp4v")
        out = cv.VideoWriter(output_path, fourcc1, fps, width_height)

        frame_counter = 0
        # Process each frame
        while True:
            ret, frame = video.capture.read()
            if not ret:
                break

            xy, r = trajectory.get_by_frame(frame_counter)

            mask = np.zeros_like(frame)
            if xy[0] is not None:
                x, y = xy
                x, y, r = int(x), int(y), int(r)
                mask = cv.circle(mask, (x, y), int(r) + 10, (255, 255, 255), -1)

            frame = cv.bitwise_and(frame, mask)

            if visualization:
                frame_to_show = cv.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)
                cv.imshow(Environment.CV_VISUALIZATION_NAME, frame_to_show)
                cv.waitKey(1)

            out.write(frame)

            frame_counter += 1

        out.release()

class  Ball_Tracker_YOLO(Ball_Tracker):
    def execute(self, params: dict):
        try:
            save_path = params.get("save_path")
        except Exception as _:
            raise Exception("Missing required parameter : save_path")
        try:
            visualization = params.get("visualization", Environment.visualization)
        except AttributeError as _:
            visualization = Environment.visualization

        tracking_results = []
        model = YOLO("./resources/models/yolov8l.pt")

        for i, view in enumerate(Environment.get_views()):
            view.trajectory = self.__track_ball(model, view.video, visualization)
            tracking_results.append(
                {"name": view.camera.name, "trajectory": view.trajectory}
            )

        # Save the final results
        DataManager.save(tracking_results, self.save_name)

    
    def __track_ball(self, model ,video : Video,visualization) -> Ball_Trajectory_2D:
        video.capture.set(cv.CAP_PROP_POS_FRAMES, 0)
        tot_frames = int(video.capture.get(cv.CAP_PROP_FRAME_COUNT))

        last_detected_centre = None
        old_gray = None
        mask = None
        trajectory = Ball_Trajectory_2D(tot_frames)  # Store all centers
        last_box = None  # Last detected bounding box
        CROP_SCALE = 20
        frame_idx = 0

        # Optical flow parameters
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        while True:
            ret, frame = video.capture.read()
            if not ret:
                break

            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Decide YOLO crop
            if last_box is None:
                yolo_frame = frame
                x1_offset, y1_offset = 0, 0
            else:
                x1, y1, x2, y2 = last_box
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                w = int((x2 - x1) * CROP_SCALE)
                h = int((y2 - y1) * CROP_SCALE)
                x1_crop = max(cx - w//2, 0)
                y1_crop = max(cy - h//2, 0)
                x2_crop = min(cx + w//2, frame.shape[1])
                y2_crop = min(cy + h//2, frame.shape[0])
                yolo_frame = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                yolo_frame = cv.resize(yolo_frame, (640, 640))
                x1_offset, y1_offset = x1_crop, y1_crop

            # Run YOLO
            results = model(yolo_frame, conf=0.05, classes=[32], augment = True)[0]

            ball_boxes = []
            for cls, box in zip(results.boxes.cls, results.boxes.xyxy):
                # Scale coordinates back to original frame
                if last_box is not None:
                    scale_x = (x2_crop - x1_crop) / yolo_frame.shape[1]
                    scale_y = (y2_crop - y1_crop) / yolo_frame.shape[0]
                    x1b, y1b, x2b, y2b = box
                    x1b = int(x1b * scale_x + x1_offset)
                    y1b = int(y1b * scale_y + y1_offset)
                    x2b = int(x2b * scale_x + x1_offset)
                    y2b = int(y2b * scale_y + y1_offset)
                    ball_boxes.append([x1b, y1b, x2b, y2b])
                else:
                    ball_boxes.append(list(map(int, box)))

            if visualization:
                vis_frame = frame.copy()
                for b in ball_boxes:
                    x1, y1, x2, y2 = b
                    cv.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                trajectory.plot_onto(vis_frame)
                frame_to_plot = cv.resize(
                    vis_frame, dsize=(0, 0), fx=0.6, fy=0.6
                )
                frame_to_plot = cv.putText(
                    frame_to_plot,
                    str(frame_idx),
                    (5, 30),
                    cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 0),
                    thickness=1,
                    lineType=cv.LINE_AA,
                )
                cv.imshow(Environment.CV_VISUALIZATION_NAME, frame_to_plot)
                cv.waitKey(2)

            # Update tracking
            if len(ball_boxes) > 0:
                x1, y1, x2, y2 = ball_boxes[0]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                radius = max(x2 - x1, y2 - y1) // 2
                last_detected_centre = np.array([[[cx, cy]]], dtype=np.float32)
                old_gray = frame_gray.copy()
                last_box = [x1, y1, x2, y2]

                trajectory.set_by_frame(np.array([cx, cy]), radius, frame_idx)
            elif last_detected_centre is not None:
                # Optical flow fallback
                p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, last_detected_centre, None, **lk_params)
                if p1 is not None and st[0][0] == 1:
                    a, b = p1[0][0].astype(int)
                    last_detected_centre = p1.reshape(-1, 1, 2)
                    old_gray = frame_gray.copy()
                    trajectory.set_by_frame(np.array([a, b]), None, frame_idx)

            frame_idx += 1

        trajectory.interpolate_radiuses() #linear interpolation of missing radiuses (not detected by optical flow)
        return trajectory
        

class Ball_Localization(Pipe):
    def execute(self, params: dict):
        try:
            visualization = params.get("visualization", False)
        except Exception as _:
            visualization = Environment.visualization

        views = Environment.get_views()
        # Compute the full projection matrices of both views
        int1 = views[0].camera.intrinsic
        int2 = views[1].camera.intrinsic
        ext1 = views[0].camera.extrinsic
        ext2 = views[1].camera.extrinsic
        proj1 = int1 @ ext1
        proj2 = int2 @ ext2

        # get the tracked points
        points_cam1 = views[0].trajectory.image_points
        points_cam2 = views[1].trajectory.image_points

        # Find the point in which both trajectories exist
        start = 0
        while points_cam1[start, 0] is None or points_cam2[start, 0] is None:
            start += 1

        # Consider only the not None points and prepare them for triangulatePoints
        points1 = points_cam1[start:-1].T.astype(np.float32)
        points2 = points_cam2[start:-1].T.astype(np.float32)

        # Triangulate
        homogeneous_points = cv.triangulatePoints(proj1, proj2, points1, points2)
        points_3D = (homogeneous_points[:3] / homogeneous_points[3]).T

        # Acceping only points over the bowling lane
        lane_coords = Environment.coords["world_lane"]
        minx, miny, _ = min(lane_coords)
        maxx, maxy, _ = max(lane_coords)
        start_over = None
        end_over = None
        for frame, (x, y, _) in enumerate(points_3D):
            if (
                start_over is None and minx <= x <= maxx and miny <= y <= maxy
            ):  # into the lane x and y
                start_over = frame
            elif (
                start_over is not None
                and end_over is None
                and (x <= minx or x >= maxx or y <= miny or y >= maxy)
            ):  # out of the lane
                end_over = frame

        if start_over is None:
            raise Exception("The ball trajectory detected is outside the bowling lane")

        if (
            end_over is None
        ):  # case in which the trajectory ends inside the bowling lane
            end_over = len(points_3D)

        points_3D = points_3D[start_over:end_over]

        # Spline Interpolation of the 3D trajectory
        lam = 100
        t = np.arange(0, len(points_3D), 1)
        y1 = points_3D[:, 0]
        y2 = points_3D[:, 1]
        y3 = points_3D[:, 2]
        spl_x = make_smoothing_spline(t, y1, lam=lam)
        spl_y = make_smoothing_spline(t, y2, lam=lam)
        spl_z = make_smoothing_spline(t, y3, lam=lam)
        if visualization:
            plot_utils.plot_3d_spline_interpolation(
                t,
                points_3D[:, 0],
                points_3D[:, 1],
                points_3D[:, 2],
                spl_x(t),
                spl_y(t),
                spl_z(t),
            )

        points_3D = np.array([spl_x(t), spl_y(t), spl_z(t)]).T.reshape(-1, 3)

        # Storing a 3D Ball Trajectory
        trajectory_3d = Ball_Trajectory_3D(views[0].trajectory.n_frames)
        start_3d = start + start_over
        end_3d = start + end_over
        for i, frame in enumerate(range(start_3d, end_3d)):
            trajectory_3d.set_by_frame(points_3D[i], frame)

        # Storing the resulting trajectory
        Environment.set("3D_trajectory", trajectory_3d)
        DataManager.save(trajectory_3d, self.save_name)

        if visualization:
            ax = plot_utils.get_3d_plot("Ball Localization : 3D Visualization")
            plot_utils.bowling_lane(ax, np.array(Environment.coords["world_lane"]))
            plot_utils.trajectory(ax, Environment.get("3D_trajectory"))
            plot_utils.show()

    def load(self, params: dict):
        try:
            visualization = params.get("visualization", False)
        except Exception as _:
            visualization = Environment.visualization

        trajectory_3d = DataManager.load(self.save_name)

        Environment.set("3D_trajectory", trajectory_3d)

        if visualization:
            ax = plot_utils.get_3d_plot("Ball Localization : 3D Visualization")
            plot_utils.bowling_lane(ax, np.array(Environment.coords["world_lane"]))
            plot_utils.trajectory(ax, Environment.get("3D_trajectory"))
            plot_utils.show()

    def plotly_page(self, params: dict) -> dict[str, html.Div]:
        trajectory_3d = DataManager.load(self.save_name)

        def makesphere(center, radius, resolution=10):
            x, y, z = center
            u, v = np.mgrid[
                0 : 2 * np.pi : resolution * 2j, 0 : np.pi : resolution * 1j
            ]
            X = radius * np.cos(u) * np.sin(v) + x
            Y = radius * np.sin(u) * np.sin(v) + y
            Z = radius * np.cos(v) + z
            return go.Surface(x=X, y=Y, z=Z, opacity=0.6)

        xyz_t = trajectory_3d.get_coords()
        xt = xyz_t[:, 0]
        yt = xyz_t[:, 1]
        zt = xyz_t[:, 2]

        lane_pos = np.array(Environment.coords["world_lane"])
        x = lane_pos[:, 0]
        y = lane_pos[:, 1]
        z = lane_pos[:, 2]

        # TODO compute actual radius
        r = (21.83 / 2) / 100

        lane = go.Figure(
            data=[
                go.Mesh3d(
                    x=x, y=y, z=z, color="lightblue", opacity=0.8, name="Bowling Lane"
                ),  # lane
                go.Scatter3d(
                    x=x[1:3],
                    y=y[1:3],
                    z=z[1:3],
                    mode="lines",
                    name="Pit",
                    line=dict(width=5, color="red"),
                ),  # end of the bowling lane
                go.Scatter3d(
                    x=xt,
                    y=yt,
                    z=zt,
                    mode="lines",
                    name="Trajectory",
                    line=dict(width=5, color="green"),
                ),  # end of the bowling lane
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
            frames=[go.Frame(data=makesphere(pos, r)) for pos in xyz_t],
        )

        lane.update_scenes(aspectmode="data")

        graph = dcc.Graph(figure=lane, style={"width": "100%", "height": "95vh"})

        page = html.Div(children=graph)

        return {self.__class__.__name__: page}