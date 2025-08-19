import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import math


class RotationAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open video file {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.angular_speeds = []
        self.timestamps = []
        self.ball_centers = []
        self.ball_radii = []
        self.smoothing_window = 7
        self.speed_buffer = deque(maxlen=self.smoothing_window)
        self.feature_params = dict(maxCorners=100, qualityLevel=0.0001, minDistance=10, blockSize=7)
        self.lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.p0 = None
        self.current_ball_center = None
        self.current_ball_radius = None
        self.initial_ball_area_estimate = None
        self.min_circularity = 0.7
        self.min_aspect_ratio = 0.8
        self.max_aspect_ratio = 1.2
        self.ransac_threshold = 5.0
        self.ransac_confidence = 0.99

    def detect_ball_center(self, frame, prev_center=None, prev_radius=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        potential_balls = []
        min_area_bound = 0
        max_area_bound = float('inf')
        if self.initial_ball_area_estimate is not None:
            min_area_bound = max(self.initial_ball_area_estimate * 0.1, 100)
            max_area_bound = min(self.initial_ball_area_estimate * 2.5, frame.shape[0] * frame.shape[1] * 0.2)
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (min_area_bound < area < max_area_bound):
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            if radius == 0:
                continue
            circle_area = math.pi * (radius ** 2)
            circularity = area / circle_area if circle_area > 0 else 0
            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(contour)
            aspect_ratio = float(w_rect) / h_rect if h_rect > 0 else 0
            if circularity > self.min_circularity and self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio:
                potential_balls.append({'center': center, 'radius': radius, 'area': area})
        if not potential_balls:
            return None, None
        if prev_center is not None:
            potential_balls.sort(key=lambda b: (np.linalg.norm(np.array(b['center']) - np.array(prev_center)), -b['area']))
        else:
            potential_balls.sort(key=lambda b: -b['area'])
        best_ball = potential_balls[0]
        return best_ball['center'], best_ball['radius']

    def initialize_feature_tracking(self, frame, center, radius):
        mask = np.zeros_like(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        tight_radius = int(radius * 0.6)
        cv2.circle(mask, center, tight_radius, 255, -1)
        p0 = cv2.goodFeaturesToTrack(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask=mask, **self.feature_params)
        if p0 is not None:
            filtered = []
            for pt in p0:
                x, y = pt[0]
                if np.linalg.norm(np.array([x, y]) - np.array(center)) <= radius * 0.65:
                    filtered.append([[x, y]])
            p0 = np.array(filtered, dtype=np.float32) if filtered else None
        self.p0 = p0
        return p0

    def track_ball_klt(self, prev_frame, curr_frame, prev_points):
        if prev_points is None or len(prev_points) < 5:
            return None, None, None, None
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None, **self.lk_params)
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = prev_points[st == 1]
            if len(good_new) >= 5:
                M, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_threshold, maxIters=2000, confidence=self.ransac_confidence)
                if M is not None:
                    inlier_mask = inliers.flatten().astype(bool)
                    good_new_inliers = good_new[inlier_mask]
                    good_old_inliers = good_old[inlier_mask]
                    if len(good_new_inliers) < 5:
                        self.p0 = None
                        return None, None, None, None
                    angular_displacement_rad = math.atan2(M[1, 0], M[0, 0])
                    center_shift = np.mean(good_new_inliers - good_old_inliers, axis=0)
                    new_ball_center_x = int(self.current_ball_center[0] + center_shift[0])
                    new_ball_center_y = int(self.current_ball_center[1] + center_shift[1])
                    scale_factor_x = np.linalg.norm(M[:, 0])
                    scale_factor_y = np.linalg.norm(M[:, 1])
                    avg_scale_factor = (scale_factor_x + scale_factor_y) / 2
                    new_ball_radius = int(self.current_ball_radius * avg_scale_factor)
                    self.p0 = good_new_inliers.reshape(-1, 1, 2)
                    return (new_ball_center_x, new_ball_center_y), new_ball_radius, angular_displacement_rad, good_new_inliers
        self.p0 = None
        return None, None, None, None

    def smooth_angular_speed(self, raw_speed):
        self.speed_buffer.append(raw_speed if raw_speed is not None else None)
        valid_speeds = [s for s in self.speed_buffer if s is not None]
        if len(valid_speeds) < 3:
            return raw_speed if raw_speed is not None else (valid_speeds[-1] if valid_speeds else None)
        median_speed = np.median(valid_speeds)
        if raw_speed is not None and abs(raw_speed - median_speed) > (median_speed * 0.7 + 5):
            return median_speed
        return np.mean(valid_speeds)

    def is_speed_valid(self, speed):
        return speed is not None and not np.isnan(speed) and not np.isinf(speed) and 0 <= speed <= 200

    def find_first_ball_frame(self, max_search_frames=300):
        frame_number = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while frame_number < min(max_search_frames, self.frame_count):
            ret, frame = self.cap.read()
            if not ret:
                break
            ball_center, ball_radius = self.detect_ball_center(frame)
            if ball_center is not None and ball_radius is not None:
                self.initial_ball_area_estimate = math.pi * (ball_radius ** 2)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                return frame_number, frame, ball_center, ball_radius
            frame_number += 1
        return None, None, None, None

    def analyze_video(self):
        start_frame_num, prev_frame, initial_center, initial_radius = self.find_first_ball_frame()
        if start_frame_num is None:
            print("Error: Could not detect ball in initial frames. Exiting.")
            return
        self.current_ball_center = initial_center
        self.current_ball_radius = initial_radius
        print(f"Starting analysis from frame {start_frame_num}. Initial center: {initial_center}, radius: {initial_radius}")
        print(f"Video FPS: {self.fps}")
        self.p0 = self.initialize_feature_tracking(prev_frame, initial_center, initial_radius)
        frame_number = start_frame_num + 1
        dt = 1.0 / self.fps
        while True:
            ret, curr_frame = self.cap.read()
            if not ret:
                break
            new_center = None
            new_radius = None
            angular_displacement_rad = None
            tracked_features_vis = None
            if self.p0 is not None and len(self.p0) >= 5:
                new_center_klt, new_radius_klt, angular_disp, tracked_features_vis = self.track_ball_klt(prev_frame, curr_frame, self.p0)
                if tracked_features_vis is not None and len(tracked_features_vis) < 5:
                    self.p0 = self.initialize_feature_tracking(curr_frame, self.current_ball_center, self.current_ball_radius)
                    new_center_klt, new_radius_klt, angular_disp, tracked_features_vis = self.track_ball_klt(prev_frame, curr_frame, self.p0)
                if new_center_klt is not None:
                    new_center = new_center_klt
                    new_radius = new_radius_klt
                    angular_displacement_rad = angular_disp
            if new_center is None or new_radius is None:
                new_center_contour, new_radius_contour = self.detect_ball_center(curr_frame, self.current_ball_center, self.current_ball_radius)
                if new_center_contour is not None:
                    new_center = new_center_contour
                    new_radius = new_radius_contour
                    self.current_ball_center = new_center
                    self.current_ball_radius = new_radius
                    self.p0 = self.initialize_feature_tracking(curr_frame, new_center, new_radius)
                    angular_displacement_rad = None
            if tracked_features_vis is not None:
                cv2.putText(curr_frame, f"Tracked features: {len(tracked_features_vis)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                for point in tracked_features_vis:
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(curr_frame, (x, y), 3, (0, 255, 0), -1)
            if new_center is not None and new_radius is not None:
                self.current_ball_center = new_center
                self.current_ball_radius = new_radius
                if angular_displacement_rad is not None:
                    raw_angular_speed = abs(angular_displacement_rad) / dt
                    if self.is_speed_valid(raw_angular_speed):
                        smoothed_speed = self.smooth_angular_speed(raw_angular_speed)
                        if smoothed_speed is not None:
                            self.angular_speeds.append(smoothed_speed)
                            self.timestamps.append((frame_number - start_frame_num) / self.fps)
                            self.ball_centers.append(self.current_ball_center)
                            self.ball_radii.append(self.current_ball_radius)
                            if frame_number % 10 == 0:
                                print(f"Frame {frame_number}: Center={self.current_ball_center}, Radius={self.current_ball_radius}, Speed={smoothed_speed:.2f} rad/s")
                        else:
                            self.angular_speeds.append(None)
                            self.timestamps.append((frame_number - start_frame_num) / self.fps)
                            self.ball_centers.append(self.current_ball_center)
                            self.ball_radii.append(self.current_ball_radius)
                    else:
                        self.angular_speeds.append(None)
                        self.timestamps.append((frame_number - start_frame_num) / self.fps)
                        self.ball_centers.append(self.current_ball_center)
                        self.ball_radii.append(self.current_ball_radius)
                else:
                    self.angular_speeds.append(None)
                    self.timestamps.append((frame_number - start_frame_num) / self.fps)
                    self.ball_centers.append(self.current_ball_center)
                    self.ball_radii.append(self.current_ball_radius)
            else:
                self.angular_speeds.append(None)
                self.timestamps.append((frame_number - start_frame_num) / self.fps)
                self.ball_centers.append(None)
                self.ball_radii.append(None)
            vis_frame = curr_frame.copy()
            if self.current_ball_center is not None and self.current_ball_radius is not None:
                cv2.circle(vis_frame, self.current_ball_center, int(self.current_ball_radius), (0, 255, 0), 2)
                cv2.circle(vis_frame, self.current_ball_center, 4, (0, 0, 255), -1)
                if tracked_features_vis is not None:
                    for point in tracked_features_vis:
                        x, y = point.ravel().astype(int)
                        cv2.circle(vis_frame, (x, y), 3, (255, 0, 0), -1)
            if self.angular_speeds and self.angular_speeds[-1] is not None:
                text = f"Angular Speed: {self.angular_speeds[-1]:.2f} rad/s (Radius: {self.current_ball_radius})"
                cv2.putText(vis_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(vis_frame, "Angular Speed: N/A", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Bowling Ball Analysis", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            prev_frame = curr_frame.copy()
            frame_number += 1
        self.cap.release()
        cv2.destroyAllWindows()

    def plot_results(self):
        valid_angular_speeds = [speed for speed in self.angular_speeds if speed is not None]
        valid_timestamps_speed = [self.timestamps[i] for i, speed in enumerate(self.angular_speeds) if speed is not None]

        if not valid_angular_speeds:
            print("No valid data")
            return

        plt.figure(figsize=(12, 4))

        plt.plot(valid_timestamps_speed, valid_angular_speeds, 'b-', linewidth=2)
        plt.title('Angular speed over time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Angular speed (rad/s)')
        plt.grid(True, alpha=0.3)

        plt.show()

        print(f"\nResults Summary (Angular Speed):")
        print(f"Average angular speed: {np.mean(valid_angular_speeds):.2f} ± {np.std(valid_angular_speeds):.2f} rad/s")
        print(f"Speed range: {np.min(valid_angular_speeds):.2f} to {np.max(valid_angular_speeds):.2f} rad/s")
        print(f"Total valid measurements: {len(valid_angular_speeds)}")

if __name__ == "__main__":
    video_path = "resources/videos/tracking/nothing_2a/default_test.mp4"
    analyzer = RotationAnalyzer(video_path)
    print("Starting bowling ball analysis with improved KLT and RANSAC...")
    analyzer.analyze_video()
    print("Analysis complete. Plotting results...")
    analyzer.plot_results()