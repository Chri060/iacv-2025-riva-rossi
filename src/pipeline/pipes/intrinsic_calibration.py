from collections.abc import Iterable
import glob, os, random
import cv2 as cv
import numpy as np
from cv2.typing import MatLike
from typing import Tuple, cast

from pipeline.environment import DataManager, Environment
from pipeline.pipe import Pipe


class IntrinsicCalibration(Pipe):
    """
    Performs intrinsic camera calibration using checkerboard images or video frames.

    This pipeline stage:
        1. Loads checkerboard images and video frames for each camera.
        2. Detects and refines checkerboard corners for sub-pixel accuracy.
        3. Computes intrinsic camera matrices and distortion coefficients.
        4. Updates Environment cameras with calibration results.
        5. Optionally visualizes checkerboard detections during calibration.
        6. Saves calibration results via DataManager.
    """

    def execute(self, params: dict):
        """
        Executes intrinsic calibration for all cameras in the Environment.

        Args:
            params (dict): Configuration parameters containing:
                images_path (str): Path to images and videos for calibration.
                visualization (bool, optional): Whether to visualize checkerboard detection.
                    Defaults to Environment.visualization.
                checkerboard_sizes (list[list[int]], optional): Checkerboard sizes for each camera.
                    Defaults to [[9, 6], [9, 6]].
        """

        # Load parameters
        images_path = params["images_path"]
        visualization = params.get("visualization", Environment.visualization)
        checkerboard_sizes = params.get("checkerboard_sizes", [[9, 6], [9, 6]])

        calibration_results = []
        for i, camera_name in enumerate(Environment.camera_names):
            checkerboard_size = cast(Tuple[int, int], tuple(checkerboard_sizes[i]))

            # Stopping criteria for corner refinement (sub-pixel accuracy)
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            # Generate "world points" based on checkerboard grid (Z=0 plane)
            world_points = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
            world_points[:, :2] = np.mgrid[0: checkerboard_size[0], 0: checkerboard_size[1]].T.reshape(-1, 2)

            # Storage for calibration
            object_points: list[np.ndarray] = []
            image_points: list[np.ndarray] = []
            img_shape: tuple[int, int] | None = None

            # Find all images (*.jpg) for this camera
            images = glob.glob(os.path.join(images_path, camera_name, "*.jpg"))
            print(f"Calibrating {camera_name}...")

            # Process calibration images
            if images:
                for input_image in images:
                    img = cv.imread(input_image)

                    if img is None:
                        print(f"Error reading image: {input_image}")
                        continue

                    refined_corners = self.find_checkerboard(img, checkerboard_size, criteria, visualization)

                    if refined_corners is not None:
                        object_points.append(world_points)
                        image_points.append(refined_corners)

                        img_shape = img.shape[:2]

            # Process calibration videos
            videos = glob.glob(os.path.join(images_path, camera_name, "*.mp4"))

            if videos:
                for video in videos:
                    capture = cv.VideoCapture(video)
                    total_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

                    if total_frames < 60:
                        print(f"Warning: {video} has less than 60 frames.")

                    frame_indices = sorted(random.sample(range(total_frames), min(60, total_frames)))

                    for idx in frame_indices:
                        capture.set(cv.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = capture.read()

                        if not ret:
                            print(f"Unable to read frame {idx} from {video}")
                            continue

                        refined_corners = self.find_checkerboard(frame, checkerboard_size, criteria, visualization)

                        if refined_corners is not None:
                            object_points.append(world_points)
                            image_points.append(refined_corners)
                            img_shape = frame.shape[:2]

                    capture.release()

            print(f"Checkerboard matched: {len(image_points)}\n")

            # Run calibration
            if len(object_points) > 0 and img_shape is not None:

                view = Environment.get(camera_name)

                ret, mtx, dist, _, _ = cv.calibrateCamera(object_points, image_points, img_shape, cameraMatrix=None, distCoeffs=None)

                # Save results in Environment for later use
                view = Environment.get(camera_name)
                view.camera.intrinsic = mtx
                view.camera.distortion = dist
                calibration_results.append({"camera_name": camera_name, "intrinsic": mtx, "distortion": dist})
            else:
                raise Exception("No valid checkerboard detections. Calibration failed.")

        # Save results
        DataManager.save(calibration_results, self.save_name, intrinsic=True)

        # Show results
        self.show_results(calibration_results)

        input("\033[92mPress Enter to continue...\033[0m")

    def load(self):
        """
        Loads saved intrinsic calibration results from DataManager.
        """

        cal_results = cast(Iterable, DataManager.load(self.save_name, intrinsic=True))

        for res in cal_results:
            view = Environment.get(res["camera_name"])
            view.camera.intrinsic = res["intrinsic"]
            view.camera.distortion = res["distortion"]

        self.show_results(cast(Iterable, DataManager.load(self.save_name, intrinsic=True)))

        input("\033[92mPress Enter to continue...\033[0m")

    def plotly_page(self, params: dict):
        """
        Placeholder for a Plotly visualization page. Not implemented.
        """

        return None

    @staticmethod
    def show_results(cal_results: Iterable):
        """
        Prints intrinsic matrices and distortion coefficients for each camera.

        Args:
            cal_results (Iterable[dict]): Iterable of calibration result dictionaries.
                Each dictionary contains:
                    camera_name (str): Camera identifier.
                    intrinsic (np.ndarray): Intrinsic camera matrix.
                    distortion (np.ndarray): Distortion coefficients.
        """

        for res in cal_results:
            print(f"\033[96m>>>>> Camera: {res['camera_name']}\033[0m")
            print("Intrinsic matrix:")
            for row in res['intrinsic']:
                print("  " + "  ".join(f"{val:10.4f}" for val in row))

            print("Distortion coefficients:")
            print("  " + "  ".join(f"{val:10.6f}" for val in res['distortion'][0]))
            print("")

    @staticmethod
    def find_checkerboard(img: MatLike, checkerboard_size: Tuple[int, int], criteria: Tuple[int, int, float],
                          visualization: bool):
        """
        Detects a checkerboard in an image and refines corner positions.

        Args:
            img (MatLike): Input BGR image.
            checkerboard_size (Tuple[int, int]): Number of inner corners per row and column (cols, rows).
            criteria (Tuple[int, int, float]): Termination criteria for sub-pixel refinement.
            visualization (bool): If True, displays the image with detected corners.

        Returns:
            np.ndarray | None: Refined corner positions if detected; otherwise None.
        """

        # Convert to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Improve contrast using CLAHE
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Detection flags for robustness
        flags = (cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

        # Detect checkerboard corners
        ret, corners = cv.findChessboardCorners(gray, checkerboard_size, flags)

        if ret:
            # Refine corners for sub-pixel accuracy
            refined_corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Visualization
            if visualization:
                cv.drawChessboardCorners(img, checkerboard_size, refined_corners, ret)
                to_plot = cv.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)
                cv.imshow(Environment.CV_VISUALIZATION_NAME, to_plot)
                cv.waitKey(1)

            return refined_corners

        return None
