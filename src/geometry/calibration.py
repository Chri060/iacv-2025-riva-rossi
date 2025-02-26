import cv2 as cv
import numpy as np
import glob
import os
import re

def intrinsic(images_path, checkerboard_size = (9, 6), show_detection = False):
    # Termination criteria for corner sub-pixel refinement
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points based on the checkerboard grid
    world_points = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    world_points[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

    # Lists to store object points and image points from all images
    object_points = []  # 3D points in real-world space
    image_points = []  # 2D points in image plane
    images = glob.glob(os.path.join(images_path, "*.jpg"))

    # Check if any images were found
    if not images:
        print(f"No images found in the specified directory. Path : {images_path}")
        return None, None

    print(f"Found {len(images)} images. Processing...")

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
        flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE

        # Detect the checkerboard
        ret, corners = cv.findChessboardCorners(gray, checkerboard_size, flags)

        print(f"Processing: {__remove_prefix(input_image, images_path=images_path)} - Checkerboard detected: {ret}")

        if ret:
            # Store object points
            object_points.append(world_points)

            # Refine corner locations for better accuracy
            refined_corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            image_points.append(refined_corners)

            # Draw detected corners on the image for visualization
            if show_detection:
                cv.drawChessboardCorners(img, checkerboard_size, refined_corners, ret)
                cv.imshow("Detected Corners", img)
                cv.waitKey(2000)

    # Proceed with calibration if at least one checkerboard was detected
    if len(object_points) > 0:
        img_shape = gray.shape[::-1]  # Image size (width, height)
        ret, mtx, dist, _, _= cv.calibrateCamera(object_points, image_points, img_shape, None, None)
        return mtx, dist
    else:
        print("No valid checkerboard detections. Calibration failed.")
        return None, None

def intrinsic_distortion_fix(mtx, dist, img_path, show_distortion=False, corners=None, scale=0.3):
    # Read image
    img = cv.imread(img_path)

    # Compute optimal new camera matrix
    h, w = img.shape[:2]
    mtx_undist, _ = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    if show_distortion:
        # Undistort the image
        img_undist = cv.undistort(img, mtx, dist, None, newCameraMatrix=mtx_undist)

        # Define original points (ensure shape: Nx1x2)
        points = corners.astype(np.float32).reshape(-1, 1, 2)

        # Undistort points with the new camera matrix
        points_undist = cv.undistortPoints(
            points, 
            mtx, 
            dist, 
            None,  
            P=mtx_undist  # Project using new camera matrix
        ).reshape(-1, 2)

        
        # Draw lines (original and corrected)
        points = points.reshape(-1,2).astype(np.int32)
        points_undist_ts = points_undist.astype(np.int32)
        cv.line(img, tuple(points[0]), tuple(points[1]), 255, 1)
        cv.line(img_undist, tuple(points_undist_ts[0]), tuple(points_undist_ts[1]), 255, 1)

        # Combine images side by side
        combined_img = cv.hconcat([img, img_undist])

        # Display the combined image
        combined_img = cv.resize(combined_img, (0,0), fx=scale, fy=scale)
        cv.imshow("Original vs Undistorted", combined_img)
        cv.waitKey(0)  # Wait for a key press to close the window
        cv.destroyAllWindows()  # Close all OpenCV windows
    
    return mtx_undist, points_undist

def extrinsic(int, world_points, image_points):
    # Find rotation and translation vectors with PnP without distorsion
    _, rotation_vector, translation_vector = cv.solvePnP(world_points, image_points, int, None, flags=cv.SOLVEPNP_ITERATIVE)

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv.Rodrigues(rotation_vector)

    # Compute camera position in world coordinates
    camera_position = -rotation_matrix.T @ translation_vector

    # Obtaining camera orientation
    camera_orientation = rotation_matrix.T
    
    extrinsic = np.hstack((rotation_matrix, translation_vector))

    return extrinsic, camera_position, camera_orientation

# Remove prefix for better visualization
def __remove_prefix(text, images_path):
    pattern = re.escape(images_path + '/')
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text