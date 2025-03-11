import cv2 as cv
import numpy as np
import glob
import os
from pipeline.pipe import Pipe
from pipeline.environment import Environment
from pipeline.environment import DataManager


class Intrinsic_Calibration(Pipe):
    def __process_params(self, params):
        try:
            images_path = params["images_path"]
        except Exception as _:
            raise Exception("Missing required parameter : images_path")
        try: 
            visualization = params.get("visualization", Environment.visualization)
        except AttributeError as _:
            visualization = Environment.visualization
        checkerboard_sizes = params.get("checkerboard_sizes", [[9, 6], [9, 6]])
        return images_path, visualization, checkerboard_sizes

    def execute(self, params):
        images_path, visualization, checkerboard_sizes = self.__process_params(params)

        calibration_results = []
        for i, camera_name in enumerate(Environment.camera_names):
            checkerboard_size = checkerboard_sizes[i]
            # Termination criteria for corner sub-pixel refinement
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            # Prepare object points based on the checkerboard grid
            world_points = np.zeros(
                (checkerboard_size[0] * checkerboard_size[1], 3), np.float32
            )
            world_points[:, :2] = np.mgrid[
                0 : checkerboard_size[0], 0 : checkerboard_size[1]
            ].T.reshape(-1, 2)

            # Lists to store object points and image points from all images
            object_points = []  # 3D points in real-world space
            image_points = []  # 2D points in image plane
            images = glob.glob(os.path.join(images_path, camera_name, "*.jpg"))

            # Check if any images were found
            if not images:
                print(
                    f"No images found in the specified directory. Path : {images_path}"
                )
                return None, None

            print(f"Calibrating {camera_name} with {len(images)} images.")

            # Loop through each image in the folder
            for input_image in images:
                img = cv.imread(input_image)
                if img is None:
                    print(f"Error reading image: {input_image}")
                    continue

                # Convert image to grayscale
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

                # Apply histogram equalization for better contrast
                clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)

                # Define flags for the chessboard detection
                flags = (
                    cv.CALIB_CB_ADAPTIVE_THRESH
                    + cv.CALIB_CB_FAST_CHECK
                    + cv.CALIB_CB_NORMALIZE_IMAGE
                )

                # Detect the checkerboard
                ret, corners = cv.findChessboardCorners(gray, checkerboard_size, flags)

                if ret:
                    # Store object points
                    object_points.append(world_points)

                    # Refine corner locations for better accuracy
                    refined_corners = cv.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria
                    )
                    image_points.append(refined_corners)

                    # Draw detected corners on the image for visualization
                    if visualization:
                        cv.drawChessboardCorners(
                            img, checkerboard_size, refined_corners, ret
                        )
                        cv.imshow("Detected Corners", img)
                        cv.waitKey(2000)

            if visualization:
                cv.destroyAllWindows()

            # Proceed with calibration if at least one checkerboard was detected
            if len(object_points) > 0:
                img_shape = gray.shape[::-1]  # Image size (width, height)
                ret, mtx, dist, _, _ = cv.calibrateCamera(
                    object_points, image_points, img_shape, None, None
                )
                view = Environment.get(camera_name)
                view.camera.intrinsic = mtx
                view.camera.distortion = dist
                calibration_results.append(
                    {"camera_name": camera_name, "intrinsic": mtx, "distortion": dist}
                )
            else:
                raise Exception("No valid checkerboard detections. Calibration failed.")

        # Save the calibration_results at the end of the calibration process for every view
        DataManager.save(calibration_results, self.save_name)

    def load(self, params):
        cal_results = DataManager.load(self.save_name)
        for res in cal_results:
            view = Environment.get(res["camera_name"])
            view.camera.intrinsic = res["intrinsic"]
            view.camera.distortion = res["distortion"]


class Extrinsic_Calibration(Pipe):
    def execute(self, params):
        world_points = np.array(Environment.coords["world_lane"])
        ext_calibration_results = {}
        for view in Environment.get_views():
            image_points = np.array(view.lane.corners)
            intrinsic = np.array(view.camera.intrinsic)
            # Find rotation and translation vectors with PnP without distorsion
            _, rotation_vector, translation_vector = cv.solvePnP(
                world_points, image_points, intrinsic, None, flags=cv.SOLVEPNP_ITERATIVE
            )

            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv.Rodrigues(rotation_vector)

            # Compute camera position in world coordinates
            camera_position = -rotation_matrix.T @ translation_vector

            # Obtaining camera orientation
            camera_orientation = rotation_matrix.T

            extrinsic = np.hstack((rotation_matrix, translation_vector))

            # Updating Environment Camera
            view.camera.extrinsic = extrinsic
            view.camera.position = camera_position
            view.camera.rotation = camera_orientation
            ext_calibration_results.update(
                {
                    view.camera.name: {
                        "extrinsic": extrinsic,
                        "position": camera_position,
                        "rotation": camera_orientation,
                    }
                }
            )
        DataManager.save(ext_calibration_results, self.save_name)

    def load(self, params):
        ext_calibration_results = DataManager.load(self.save_name)
        for view in Environment.get_views():
            res = ext_calibration_results[view.camera.name]
            view.camera.extrinsic = res["extrinsic"]
            view.camera.position = res["position"]
            view.camera.rotation = res["rotation"]
