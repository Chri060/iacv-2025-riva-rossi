import cv2 as cv
import numpy as np
from scipy import interpolate
import math

from pipeline.pipe import Pipe
from pipeline.environment import Environment
from pipeline.environment import Ball_Trajectory_2D
from pipeline.environment import DataManager


class Lane_Detector(Pipe):
    def execute(self, params):
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
            points = self.__manual_point_selection(frame, scale=scales[i])
            capture.set(cv.CAP_PROP_POS_FRAMES, 0)
            view.lane.corners = points
            detection_results.update({view.camera.name: points})
        DataManager.save(detection_results, self.save_name)

    def load(self, params):
        try:
            visualization = params.get("visualization", Environment.visualization)
        except AttributeError as _:
            visualization = Environment.visualization

        detection_results = DataManager.load(self.save_name)
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
                        frame, (int(point[0]), int(point[1])), 2, (255, 0, 0), 3
                    )
                cv.imshow(f"Lane : {view.camera.name}", frame)
                cv.waitKey(0)
                cv.destroyAllWindows()

    def __manual_point_selection(self, image, scale=0.5):
        selected_points = []
        upscale = 1 / scale

        def select_point(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                cv.circle(image, (x, y), 3, (0, 0, 255), -1)
                cv.imshow("Image", image)
                x = x * upscale
                y = y * upscale
                selected_points.append([x, y])

        image = cv.resize(image, None, fx=scale, fy=scale)
        cv.imshow("Image", image)
        cv.setMouseCallback("Image", select_point)

        cv.waitKey(0)
        cv.destroyAllWindows()

        return selected_points


class Ball_Tracker(Pipe):
    def execute(self, params):
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
            trajectory = self.track_moving_object(
                video,
                output_path,
                roi,
                kernel_size=5,
                playback_scaling=0.6,
                visualization=visualization,
            )

            view.trajectory = trajectory
            tracking_results.append(
                {"name": view.camera.name, "trajectory": view.trajectory}
            )

        DataManager.save(tracking_results, self.save_name)

    def load(self, params):
        try:
            visualization = params.get("visualization", Environment.visualization)
        except AttributeError as _:
            visualization = Environment.visualization

        tracking_results = DataManager.load(self.save_name)
        for result in tracking_results:
            view = Environment.get(result["name"])
            view.trajectory = result["trajectory"]

            if visualization:
                while True:
                    ret, frame = view.video.capture.read()
                    if not ret:
                        break
                    view.trajectory.plot_onto(frame)
                    cv.imshow("Found trajectory for {view.camera.name}", frame)
                    cv.waitKey(1)
                cv.destroyAllWindows()

            view.video.capture.set(cv.CAP_PROP_POS_FRAMES, 0)

    def track_moving_object(
        self,
        video,
        output_path,
        roi_points,
        kernel_size=20,
        color_range=[(0, 0, 0), (255, 255, 255)],
        playback_scaling=0.5,
        visualization=False,
    ):
        if not video.capture.isOpened():
            print("Could not open the video!")
            return

        # Initialize video playback
        frame_counter = -1  # first frame is 0
        video.capture.set(cv.CAP_PROP_POS_FRAMES, 0)

        # Initialize the background subtractor
        bg_subtractor = cv.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )

        # Initialize found trajectories
        trajectories = set()

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
            frame_result_gray = cv.GaussianBlur(
                frame_result_gray, (9, 9), 2
            )  # Stronger blur to reduce noise
            circles = cv.HoughCircles(
                frame_result_gray,
                cv.HOUGH_GRADIENT,
                dp=1,  # Inverse ratio of resolution
                minDist=100,  # Minimum distance between circles
                param1=50,  # Canny edge threshold
                param2=15,  # Circle detection threshold
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
                    trajectory.plot_onto(frame_result)
                frame_to_plot = cv.resize(
                    frame_result, dsize=(0, 0), fx=playback_scaling, fy=playback_scaling
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

                cv.imshow("Localization results", frame_to_plot)
                cv.waitKey(2)

        if visualization:
            cv.destroyAllWindows()

        # Reroll video
        video.capture.set(cv.CAP_PROP_POS_FRAMES, 0)

        return self.__choose_trajectory(trajectories)

    # Returns the longest trajectory from the proposed ones
    def __choose_trajectory(self, trajectories):
        best_trajectory = None
        oldest_start = float("inf")
        for trajectory in trajectories:
            start_frame = 0
            while trajectory.get_by_frame(start_frame) is None:
                start_frame += 1
            if start_frame < oldest_start:
                oldest_start = start_frame
                best_trajectory = trajectory
        return best_trajectory

    # Updates the ongoing trajectories discarding the one that cannot be continued
    def __update_trajectories(self, curr_trajectories, circles, curr_frame, tot_frames):
        to_assign = set(curr_trajectories.copy())
        res_trajectories = set()
        while len(circles) > 0:
            best_couple = None
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
                best_couple["traj"].set_by_frame((x, y), r, curr_frame)
                res_trajectories.add(best_couple["traj"])

                # remove the trajectory
                to_assign.remove(best_couple["traj"])
                # remove the circle
                circles = circles[~np.all(circles == best_couple["circle"], axis=1)]
            else:  # if there are no more trajectories but there are still circles it creates new trajectories
                for circle in circles:
                    new_trajectory = Ball_Trajectory_2D(tot_frames)
                    x, y, r = circle
                    new_trajectory.set_by_frame((x, y), r, curr_frame)
                    res_trajectories.add(new_trajectory)
                break
        return res_trajectories

    # Creates a binary polygonal mask of a given size
    def __create_polygonal_mask(self, corners, padding, size):
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


class Ball_Localization(Pipe):
    def execute(self, params):
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
        while points_cam1[start] is None or points_cam2[start] is None:
            start += 1

        # Consider only the not None points and prepare them for triangulatePoints
        points1 = np.array(
            [[x1, y1] for (x1, y1) in points_cam1[start:-1]], dtype=np.float32
        ).T
        points2 = np.array(
            [[x2, y2] for (x2, y2) in points_cam2[start:-1]], dtype=np.float32
        ).T

        # Triangulate
        homogeneous_points = cv.triangulatePoints(proj1, proj2, points1, points2)
        points_3D = (homogeneous_points[:3] / homogeneous_points[3]).T

        Environment.set("3D_trajectory", points_3D)
        DataManager.save(points_3D, self.save_name)

    def load(self, params):
        points_3D = DataManager.load(self.save_name)
        Environment.set("3D_trajectory", points_3D)