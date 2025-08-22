import glob, os, random
import cv2 as cv
import numpy as np
from cv2.typing import MatLike
from typing import Tuple, cast

from pipeline.environment import DataManager, Environment
from pipeline.pipe import Pipe

class IntrinsicCalibration(Pipe):
    """
    This class performs intrinsic camera calibration using images or video frames
    of a checkerboard pattern. The calibration estimates the intrinsic parameters
    (camera matrix) and distortion coefficients for each camera defined in Environment.
    """

    def show_results(self):
        cal_results = DataManager.load(self.save_name)

        for res in cal_results:
            print(f"\033[96m>>>>> Camera: {res['camera_name']}\033[0m")
            print("Intrinsic matrix:")
            for row in res['intrinsic']:
                print("  " + "  ".join(f"{val:10.4f}" for val in row))

            print("Distortion coefficients:")
            print("  " + "  ".join(f"{val:10.6f}" for val in res['distortion'][0]))
            print("")

        input("\033[92mPress Enter to continue...\033[0m")

    @staticmethod
    def __process_params(params: dict):
        """
        Process and validate input parameters.
        Required:
            - images_path: path to directory containing calibration images/videos
        Optional:
            - visualization: enable OpenCV visualization (default: Environment.visualization)
            - checkerboard_sizes: list of checkerboard sizes for each camera
        """

        # Image path
        try:
            images_path = params["images_path"]
        except Exception as _:
            raise Exception("Missing required parameter : images_path")

        # Visualization flag
        try:
            visualization = params.get("visualization", Environment.visualization)
        except AttributeError as _:
            visualization = Environment.visualization

        # Checkerboard size per camera (default: 9x6 for both cameras)
        checkerboard_sizes = params.get("checkerboard_sizes", [[9, 6], [9, 6]])

        return images_path, visualization, checkerboard_sizes

    def execute(self, params: dict):
        """
        Main execution method: loops through cameras and performs intrinsic calibration.
        """

        images_path, visualization, checkerboard_sizes = self.__process_params(params)

        calibration_results = []
        for i, camera_name in enumerate(Environment.camera_names):
            checkerboard_size = cast(Tuple[int, int], tuple(checkerboard_sizes[i]))

            # Stopping criteria for corner refinement (sub-pixel accuracy)
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            # Generate "world points" based on checkerboard grid (Z=0 plane)
            world_points = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
            world_points[:, :2] = np.mgrid[0 : checkerboard_size[0], 0 : checkerboard_size[1]].T.reshape(-1, 2)

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

                    refined_corners = self.__find_checkerboard(img, checkerboard_size, criteria, visualization)

                    if refined_corners is not None:
                        object_points.append(world_points)
                        image_points.append(refined_corners)

                        img_shape = img.shape[:2]

            # Process calibration videos
            videos = glob.glob(os.path.join(images_path, camera_name, "*.mp4"))

            if videos :
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

                        refined_corners = self.__find_checkerboard(frame, checkerboard_size, criteria, visualization)

                        if refined_corners is not None:
                            object_points.append(world_points)
                            image_points.append(refined_corners)
                            img_shape = frame.shape[:2]

                    capture.release()

            print(f"Checkerboard matched: {len(image_points)}\n")

            # Run calibration
            if len(object_points) > 0 and img_shape is not None:
                ret, mtx, dist, _, _ = cv.calibrateCamera(object_points, image_points, img_shape, None, None)

                # Save results in Environment for later use
                view = Environment.get(camera_name)
                view.camera.intrinsic = mtx
                view.camera.distortion = dist
                calibration_results.append({"camera_name": camera_name, "intrinsic": mtx, "distortion": dist})
            else:
                raise Exception("No valid checkerboard detections. Calibration failed.")

        self.show_results()

        # Save results
        DataManager.save(calibration_results, self.save_name)

    @staticmethod
    def __find_checkerboard(img: MatLike, checkerboard_size: Tuple[int, int], criteria: Tuple[int, int, float], visualization: bool):
        """
        Detects a checkerboard in an image, refines corner detection,
        and optionally visualizes the result.
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

    def load(self):
        """
        Load calibration results from storage and apply them to Environment views.
        """

        cal_results = DataManager.load(self.save_name)
        for res in cal_results:
            view = Environment.get(res["camera_name"])
            view.camera.intrinsic = res["intrinsic"]
            view.camera.distortion = res["distortion"]

        self.show_results()