import cv2 as cv
import numpy as np
from dash import html
import matplotlib.pyplot as plt

from pipeline import plot_utils
from pipeline.environment import DataManager, Environment
from pipeline.pipe import Pipe

class SpinBall(Pipe):
    """
    SpinBall is a pipeline stage responsible for estimating bowling ball spin
    using optical flow on previously tracked ball trajectories.
    """

    def execute(self, params: dict):
        """
        Estimates ball spin using weighted optical flow on tracked ball regions
        and plots 2D rotation axis.
        """

        # Graph save path
        try:
            graph_save_path = params["graph_save_path"]
        except Exception as _:
            raise Exception("Missing required parameter : save_path")

        # Visualization
        try:
            visualization = params.get("visualization", Environment.visualization)
        except AttributeError:
            visualization = Environment.visualization

        smoothing_alpha = 0.2
        max_corners = 50

        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        spin_results = {}
        axis_points = {}

        for view in Environment.get_views():
            cap = view.video.capture
            fps, _, _ = view.video.get_video_properties()
            trajectory = view.trajectory
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)

            spin_rates = []
            old_gray = None
            p0 = None
            frame_idx = 0
            axis_points[view.camera.name] = []

            while True:
                ret, frame = cap.read()
                if not ret or frame_idx >= trajectory.n_frames:
                    break

                frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                vis_frame = frame.copy()
                cxcy, radius = trajectory.get_by_frame(frame_idx)

                if cxcy is not None and radius is not None:
                    cx, cy = map(int, cxcy)
                    radius = int(radius)

                    if visualization:
                        cv.circle(vis_frame, (cx, cy), radius, (0, 0, 255), 2)
                        cv.circle(vis_frame, (cx, cy), 3, (0, 0, 255), -1)

                    # Mask ball area for feature tracking
                    mask_ball = np.zeros_like(frame_gray)
                    cv.circle(mask_ball, (cx, cy), max(radius - 2, 1), 255, -1)
                    p_new = cv.goodFeaturesToTrack(frame_gray, mask=mask_ball, maxCorners=max_corners, qualityLevel=0.01, minDistance=5)

                    if p0 is not None and len(p0) > 0:
                        p1, st, _ = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                        if p1 is not None and st is not None:
                            good_new = p1[st == 1]
                            good_old = p0[st == 1]

                            filtered_new, filtered_old = [], []
                            for (new, old) in zip(good_new, good_old):
                                x_new, y_new = new.ravel()
                                if (x_new - cx) ** 2 + (y_new - cy) ** 2 <= (radius - 2) ** 2:
                                    filtered_new.append(new)
                                    filtered_old.append(old)

                            trajectory_3d = Environment.get("3D_trajectory")

                            # --- Compute 3D rotation axis & spin ---
                            axes_3d = []
                            dthetas = []
                            weights = []

                            for (new, old) in zip(filtered_new, filtered_old):
                                # Old & new displacement vectors in *image plane*
                                dx_old, dy_old = old.ravel() - [cx, cy]
                                dx_new, dy_new = new.ravel() - [cx, cy]

                                r_old_len = np.sqrt(dx_old ** 2 + dy_old ** 2)
                                if r_old_len < 0.3 * radius:
                                    continue

                                # Normalize to sphere surface in 3D
                                # Map 2D offsets into 3D local coords (z from sphere geometry)
                                z_old = np.sqrt(max(radius ** 2 - r_old_len ** 2, 0.0))
                                z_new = np.sqrt(max(radius ** 2 - (dx_new ** 2 + dy_new ** 2), 0.0))

                                r_old = np.array([dx_old, dy_old, z_old])
                                r_new = np.array([dx_new, dy_new, z_new])

                                # Rotation axis from cross product
                                axis_vec = np.cross(r_old, r_new)
                                if np.linalg.norm(axis_vec) > 1e-6:
                                    axis_vec /= np.linalg.norm(axis_vec)
                                    axes_3d.append(axis_vec)

                                # Rotation angle from dot product
                                dot = np.dot(r_old, r_new) / (np.linalg.norm(r_old) * np.linalg.norm(r_new))
                                dot = np.clip(dot, -1.0, 1.0)
                                theta = np.arccos(dot)

                                dthetas.append(theta)
                                weights.append(r_old_len / radius)

                            if dthetas:
                                avg_axis = np.mean(axes_3d, axis=0) if axes_3d else np.array([0, 0, 1])
                                avg_axis /= np.linalg.norm(avg_axis)
                            else:
                                avg_axis = np.array([0, 0, 1])

                            # Save 3D axis instead of 2D line endpoints
                            axis_points[view.camera.name].append(avg_axis)

                            if dthetas:
                                med_dtheta = np.average(dthetas, weights=weights)
                                spin_rate = med_dtheta * fps
                                prev_spin = spin_rates[-1] if spin_rates else 0.0
                                if len(spin_rates) > 0 and abs(spin_rate - prev_spin) > 20:
                                    spin_rate = prev_spin
                                spin_rate = smoothing_alpha * spin_rate + (1 - smoothing_alpha) * prev_spin
                                spin_rates.append(spin_rate)
                            else:
                                spin_rates.append(spin_rates[-1] if spin_rates else 0.0)

                            # Merge features
                            if p_new is not None:
                                filtered_new_arr = np.array(filtered_new, dtype=np.float32).reshape(-1, 1, 2)
                                p0 = np.vstack([filtered_new_arr, p_new]) if len(filtered_new) > 0 else p_new
                            else:
                                p0 = np.array(filtered_new, dtype=np.float32).reshape(-1, 1, 2) if len(
                                    filtered_new) > 0 else None
                        else:
                            p0 = p_new
                            spin_rates.append(spin_rates[-1] if spin_rates else 0.0)
                    else:
                        p0 = p_new
                        spin_rates.append(0.0)

                    old_gray = frame_gray.copy()

                    if visualization:
                        if p0 is not None:
                            for point in p0:
                                x, y = point.ravel()
                                cv.circle(vis_frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                        current_spin = spin_rates[-1] if spin_rates else 0.0
                        cv.putText(vis_frame, f"Spin: {current_spin:.1f} rad/s", (10, 30),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        frame_to_plot = cv.resize(vis_frame, dsize=(0, 0), fx=0.6, fy=0.6)
                        cv.imshow(Environment.CV_VISUALIZATION_NAME, frame_to_plot)
                        cv.waitKey(1)

                else:
                    spin_rates.append(spin_rates[-1] if spin_rates else 0.0)

                frame_idx += 1

            spin_results[view.camera.name] = np.array(spin_rates, dtype=np.float32)
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)


        DataManager.save({"spin_results": spin_results, "axis_points": axis_points}, self.save_name)
        Environment.set("spin_rates", spin_results)
        Environment.set("axis_points", axis_points)

        # --- Combine both camera results into one ---
        if len(spin_results) >= 2:
            cams = list(spin_results.keys())
            spins1 = spin_results[cams[0]]
            spins2 = spin_results[cams[1]]
            min_len = min(len(spins1), len(spins2))
            combined = []

            for i in range(min_len):
                s1, s2 = spins1[i], spins2[i]
                if s1 > 0 and s2 > 0:
                    combined.append((s1 + s2) / 2.0)
                elif s1 > 0:
                    combined.append(s1)
                elif s2 > 0:
                    combined.append(s2)
                else:
                    combined.append(0.0)

            combined = np.array(combined, dtype=np.float32)
            spin_results["combined"] = combined

        plt.figure(figsize=(10, 6))
        for cam_name, spins in spin_results.items():
            spins_rps = spins / (2 * np.pi)
            window = 5
            spins_smooth = np.convolve(spins_rps, np.ones(window) / window, mode='same')
            if cam_name == "combined":
                plt.plot(abs(spins_smooth), label="Combined", linewidth=2.5, color="black")
            else:
                plt.plot(abs(spins_smooth), label=f"{cam_name}", alpha=0.5)

        plt.xlabel("Frame")
        plt.ylabel("Spin rate (rev/s)")
        plt.title("Ball Spin Rate Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{graph_save_path}/angular_speed/{Environment.save_name}_{Environment.video_name.removesuffix(".mp4")}.png", dpi=300, bbox_inches="tight")
        if visualization:
            plt.show()
        else:
            plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Ball Spin Axes : 3D Visualization")

        # --- Plot lane & trajectory ---
        plot_utils.bowling_lane(ax, np.array(Environment.coords["world_lane"]))
        plot_utils.trajectory(ax, Environment.get("3D_trajectory"))

        # --- Plot predicted spin axes along trajectory ---
        R_ball = 0.1091  # radius [m]
        L = R_ball * 2
        trajectory_3d = Environment.get("3D_trajectory")

        cam_names = list(axis_points.keys())
        first_cam_name = cam_names[0]
        other_cam_name = cam_names[1] if len(cam_names) > 1 else None

        for i, center in enumerate(trajectory_3d.coords):
            if center[0] is None:
                continue

            # Try first camera
            axis_vec = axis_points[first_cam_name][i] if i < len(axis_points[first_cam_name]) else None

            # If not available, fallback to second camera
            if (axis_vec is None or np.linalg.norm(axis_vec) < 1e-6) and other_cam_name is not None:
                if i < len(axis_points[other_cam_name]):
                    axis_vec = axis_points[other_cam_name][i]

            # Skip if still invalid
            if axis_vec is None or np.linalg.norm(axis_vec) < 1e-6:
                continue

            axis_vec = axis_vec / np.linalg.norm(axis_vec)
            start = np.array(center) - axis_vec * L
            end = np.array(center) + axis_vec * L
            ax.plot([start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    color='red', alpha=0.4)

        # --- Define standard views (elev, azim) ---
        views = {
            "back": (20, 180),
            "top": (90, -90),
            "side": (0, 90),
            "front": (20, 0),
        }

        # --- Save from each view ---
        for name, (elev, azim) in views.items():
            ax.view_init(elev=elev, azim=azim)
            plt.savefig(
                f"{graph_save_path}/axis/{Environment.save_name}_{name}_{Environment.video_name.removesuffix(".mp4")}.png",
                dpi=300, bbox_inches="tight")
        if visualization:
            plt.show()
        else:
            plt.close(fig)

    def load(self, params: dict = None):
        """
        Load previously saved spin data and visualize.

        Args:
            params (dict, optional): Dictionary that may contain flags, e.g., 'visualization' (bool)

        Returns:
            dict: A dictionary with 'spin_results' and 'axis_points'
        """

        # Visualization
        try:
            visualization = params.get("visualization", Environment.visualization)
        except AttributeError:
            visualization = Environment.visualization

        data = DataManager.load(self.save_name)
        spin_results = data.get("spin_results")
        axis_points = data.get("axis_points")

        if spin_results is None or axis_points is None:
            raise ValueError(f"No valid spin_results or axis_points found in {self.save_name}")

        Environment.set("spin_rates", spin_results)
        Environment.set("axis_points", axis_points)

        # --- Combine both camera results into one if possible ---
        if visualization:
            if len(spin_results) >= 2:
                cams = list(spin_results.keys())
                spins1 = spin_results[cams[0]]
                spins2 = spin_results[cams[1]]
                min_len = min(len(spins1), len(spins2))
                combined = []

                for i in range(min_len):
                    s1, s2 = spins1[i], spins2[i]
                    if s1 > 0 and s2 > 0:
                        combined.append((s1 + s2) / 2.0)
                    elif s1 > 0:
                        combined.append(s1)
                    elif s2 > 0:
                        combined.append(s2)
                    else:
                        combined.append(0.0)

                combined = np.array(combined, dtype=np.float32)
                spin_results["combined"] = combined

            # --- Plot spin results ---
            plt.figure(figsize=(10, 6))
            for cam_name, spins in spin_results.items():
                spins_rps = spins / (2 * np.pi)  # convert rad/s → rev/s
                window = 5
                spins_smooth = np.convolve(spins_rps, np.ones(window) / window, mode='same')
                if cam_name == "combined":
                    plt.plot(abs(spins_smooth), label="Combined", linewidth=2.5, color="black")
                else:
                    plt.plot(abs(spins_smooth), label=f"{cam_name}", alpha=0.5)

            plt.xlabel("Frame")
            plt.ylabel("Spin rate (rev/s)")
            plt.title("Ball Spin Rate Over Time")
            plt.legend()
            plt.grid(True)
            plt.show()

            # --- Plot 3D lane, trajectory & spin axes ---
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title("Ball Spin Axes : 3D Visualization")

            # Lane & trajectory
            plot_utils.bowling_lane(ax, np.array(Environment.coords["world_lane"]))
            trajectory_3d = Environment.get("3D_trajectory")
            plot_utils.trajectory(ax, trajectory_3d)

            # Spin axes
            R_ball = 0.11  # m
            L = R_ball * 2
            for center, axis_vec in zip(trajectory_3d.coords, axis_points[list(axis_points.keys())[0]]):
                if center[0] is None:
                    continue
                if axis_vec is None or np.linalg.norm(axis_vec) < 1e-6:
                    continue
                axis_vec = axis_vec / np.linalg.norm(axis_vec)
                start = np.array(center) - axis_vec * L
                end = np.array(center) + axis_vec * L
                ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='red', alpha=0.4)

            plt.show()

        input("\033[92mPress Enter to continue...\033[0m")

    def plotly_page(self, params: dict) -> dict[str, html.Div]:
        return