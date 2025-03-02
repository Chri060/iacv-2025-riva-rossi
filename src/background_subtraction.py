import cv2
import numpy as np
import data.coords as coords


def process_video(background_path, video_path, output_path, roi_points, tolerance=15):
    # Load the background image
    background = cv2.imread(background_path)
    if background is None:
        print("Could not load background image!")
        return

    # Open the video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Could not open the video!")
        return

    # Get video properties
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Define the ROI bounding box with tolerance
    x_min = int(min(roi_points[:, 0]) - tolerance)  # Ensure x_min is an integer
    x_max = int(max(roi_points[:, 0]) + tolerance)  # Ensure x_max is an integer
    y_min = int(min(roi_points[:, 1]) - tolerance)  # Ensure y_min is an integer
    y_max = int(max(roi_points[:, 1]) + tolerance)  # Ensure y_max is an integer

    # Create Shi-Tomasi corner detector parameters for feature detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Initialize the previous frame and points for optical flow
    ret, prev_frame = video.read()
    if not ret:
        print("Could not read the first frame.")
        return
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Detect features in the first frame
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    # Process each frame
    while True:
        ret, frame = video.read()

        # Check if a frame was found
        if not ret:
            break

        # Convert frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow to track features
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_pts, None)

        # Select good points based on status
        good_new = next_pts[status == 1]
        good_old = prev_pts[status == 1]

        # Estimate the motion using the good points
        if len(good_new) > 0:
            # Calculate the transformation matrix to align the frames
            transformation_matrix, _ = cv2.estimateAffinePartial2D(good_old, good_new)

            # Warp the frame to stabilize it
            stabilized_frame = cv2.warpAffine(frame, transformation_matrix, (frame_width, frame_height))

        else:
            stabilized_frame = frame  # Fallback in case no features are found

        # Compute absolute difference between stabilized frame and background
        fg_mask = cv2.absdiff(stabilized_frame, background)

        # Apply Gaussian blur to smooth the image
        fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)

        # Threshold the difference
        threshold_value = 15
        _, fg_mask = cv2.threshold(fg_mask, threshold_value, 255, cv2.THRESH_BINARY)

        # Create a mask based on the ROI specified before
        roi_mask = np.zeros_like(fg_mask, dtype=np.uint8)
        roi_mask[y_min:y_max, x_min:x_max] = 255

        # Combine the foreground mask and the ROI mask
        fg_mask = cv2.bitwise_and(fg_mask, roi_mask)

        # Convert the final mask to 3 channels (only if it's a single channel image)
        # if len(fg_mask.shape) == 2 or fg_mask.shape[2] == 1:
        #    fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

        # Apply mask to the stabilized frame
        result = cv2.bitwise_and(stabilized_frame, fg_mask)

        # Write the processed frame to output video
        out.write(result)

        # Update previous frame and points
        prev_gray = frame_gray.copy()
        prev_pts = good_new.reshape(-1, 1, 2)

    # Release resources
    video.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete. Video saved at:", output_path)


if __name__ == "__main__":
    background_path = "resources/localization/lumix/lane.png"
    video_path = "resources/video/lumix/opt_7.MP4"
    output_path = "resources/video_modified/mod.mp4"
    process_video(background_path, video_path, output_path, coords.LUM_LANE_CORNERS)