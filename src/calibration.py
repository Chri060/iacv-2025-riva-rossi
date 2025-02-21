import cv2
import numpy as np
import glob
import os
import re

def intrinsic(images_path, checkerboard_size = (9, 6), square_size = 25, show_detection = False):
    # Termination criteria for corner sub-pixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points based on the checkerboard grid
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

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
        img = cv2.imread(input_image)
        if img is None:
            print(f"Error reading image: {input_image}")
            continue

        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Define flags for the chessboard detection
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE

        # Detect the checkerboard
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, flags)

        print(f"Processing: {__remove_prefix(input_image, images_path=images_path)} - Checkerboard detected: {ret}")

        if ret:
            # Store object points
            object_points.append(objp)

            # Refine corner locations for better accuracy
            refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            image_points.append(refined_corners)

            # Draw detected corners on the image for visualization
            cv2.drawChessboardCorners(img, checkerboard_size, refined_corners, ret)
            if show_detection:
                cv2.imshow("Detected Corners", img)
                cv2.waitKey(1000)

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    # Proceed with calibration if at least one checkerboard was detected
    if len(object_points) > 0:
        img_shape = gray.shape[::-1]  # Image size (width, height)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img_shape, None, None)
        return mtx, dist
    else:
        print("No valid checkerboard detections. Calibration failed.")
        return None, None


# Remove prefix for better visualization
def __remove_prefix(text, images_path):
    pattern = re.escape(images_path + '\\')
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text